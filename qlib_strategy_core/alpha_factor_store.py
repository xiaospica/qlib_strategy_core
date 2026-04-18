from __future__ import annotations

import gc
import json
import os
import shutil
import threading
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd


STORE_SCHEMA_VERSION = "1.0"

_factor_memory_cache: Dict[str, pd.DataFrame] = {}
_cache_time_range: Dict[str, Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]] = {}
_cache_lock = threading.Lock()
_cache_stats = {"hits": 0, "misses": 0}


def clear_factor_memory_cache():
    """清除因子数据的内存缓存"""
    global _factor_memory_cache, _cache_time_range, _cache_stats
    with _cache_lock:
        _factor_memory_cache.clear()
        _cache_time_range.clear()
        _cache_stats = {"hits": 0, "misses": 0}
    gc.collect()
    print("[内存缓存] 因子数据缓存已清除")


def get_cache_stats():
    """获取缓存统计信息"""
    with _cache_lock:
        return {
            "hits": _cache_stats["hits"],
            "misses": _cache_stats["misses"],
            "cached_items": len(_factor_memory_cache),
        }


def _to_ts(value: Union[str, pd.Timestamp, None]) -> Optional[pd.Timestamp]:
    if value is None:
        return None
    return pd.Timestamp(value)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    try:
        with tmp_path.open("w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _store_dir(
    store_root: Union[str, Path],
    alpha_family: str,
    market: str,
    freq: str,
) -> Path:
    return Path(store_root).expanduser().resolve() / alpha_family.lower() / market.lower() / str(freq).lower()


def _metadata_path(
    store_root: Union[str, Path],
    alpha_family: str,
    market: str,
    freq: str,
) -> Path:
    return _store_dir(store_root, alpha_family, market, freq) / "metadata.json"


def _normalize_factor_df(df_factor: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df_factor.index, pd.MultiIndex):
        raise ValueError("df_factor index must be MultiIndex(datetime, instrument)")
    if "datetime" not in df_factor.index.names or "instrument" not in df_factor.index.names:
        raise ValueError("df_factor index names must include datetime and instrument")
    return df_factor.sort_index()


def _extract_partition_dates(df_factor: pd.DataFrame) -> pd.Series:
    dt = pd.DatetimeIndex(df_factor.index.get_level_values("datetime"))
    return dt.normalize().strftime("%Y-%m-%d")


def _list_partitions(store_dir: Path) -> List[str]:
    dates: List[str] = []
    if not store_dir.exists():
        return dates
    for path in store_dir.glob("datetime=*"):
        if path.is_dir() and "=" in path.name:
            dates.append(path.name.split("=", 1)[1])
    dates.sort()
    return dates


def _build_metadata(
    alpha_family: str,
    market: str,
    freq: str,
    features: Sequence[str],
    partitions: Sequence[str],
    row_count: int,
    config_hash: Optional[str] = None,
) -> Dict[str, object]:
    return {
        "schema_version": STORE_SCHEMA_VERSION,
        "alpha_family": alpha_family.lower(),
        "market": market.lower(),
        "freq": str(freq).lower(),
        "features": list(features),
        "partition_count": len(partitions),
        "row_count": int(row_count),
        "start_date": partitions[0] if partitions else None,
        "end_date": partitions[-1] if partitions else None,
        "config_hash": config_hash,
        "partitions": list(partitions),
    }


def read_factor_store_metadata(
    store_root: Union[str, Path],
    alpha_family: str,
    market: str,
    freq: str,
) -> Dict[str, object]:
    meta_path = _metadata_path(store_root, alpha_family, market, freq)
    if not meta_path.exists():
        raise FileNotFoundError(f"因子库元数据不存在: {meta_path}")
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_factor_store(
    store_root: Union[str, Path],
    alpha_family: str,
    market: str,
    freq: str,
    required_features: Sequence[str],
    start_time: Union[str, pd.Timestamp, None] = None,
    end_time: Union[str, pd.Timestamp, None] = None,
) -> Dict[str, object]:
    meta = read_factor_store_metadata(store_root, alpha_family, market, freq)
    feature_set = set(meta.get("features", []))
    missing_features = [f for f in required_features if f not in feature_set]
    if missing_features:
        raise ValueError(f"因子库缺少特征列: {missing_features[:10]}")

    partitions = list(meta.get("partitions", []))
    if not partitions:
        raise ValueError("因子库无可用分区")

    st = _to_ts(start_time)
    et = _to_ts(end_time)
    if st is not None and et is not None and st > et:
        raise ValueError("start_time must be <= end_time")

    if st is not None:
        st_date = st.normalize().strftime("%Y-%m-%d")
        if st_date < str(meta.get("start_date")):
            raise ValueError(f"因子库起始日期不足: request={st_date}, store_start={meta.get('start_date')}")
    if et is not None:
        et_date = et.normalize().strftime("%Y-%m-%d")
        if et_date > str(meta.get("end_date")):
            print(f'因子库结束日期不足: request={et_date}, store_end={meta.get("end_date")}')
            # raise ValueError(f"因子库结束日期不足: request={et_date}, store_end={meta.get('end_date')}")
    return meta


def build_factor_store(
    df_factor: pd.DataFrame,
    store_root: Union[str, Path],
    alpha_family: str,
    market: str,
    freq: str,
    overwrite: bool = False,
    config_hash: Optional[str] = None,
    verbose: bool = False,
    progress_every: int = 20,
) -> Dict[str, object]:
    df_factor = _normalize_factor_df(df_factor)
    target_dir = _store_dir(store_root, alpha_family, market, freq)
    if target_dir.exists() and overwrite:
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    feature_names = [str(c) for c in df_factor.columns]
    df_data = df_factor.copy()
    df_data["__partition_date__"] = _extract_partition_dates(df_data)
    df_data = df_data.reset_index()
    grouped = df_data.groupby("__partition_date__", sort=True)
    total_parts = grouped.ngroups
    if verbose:
        print(
            f"[factor_store] 开始写入: family={alpha_family}, market={market}, freq={freq}, "
            f"rows={len(df_factor)}, partitions={total_parts}"
        )

    for i, (part_date, part_df) in enumerate(grouped, start=1):
        out_dir = target_dir / f"datetime={part_date}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "data.parquet"
        write_df = part_df.drop(columns=["__partition_date__"])
        write_df.to_parquet(out_path, index=False)
        if verbose and (i == 1 or i % max(progress_every, 1) == 0 or i == total_parts):
            print(f"[factor_store] 写入进度: {i}/{total_parts}, datetime={part_date}, rows={len(write_df)}")

    partitions = _list_partitions(target_dir)
    metadata = _build_metadata(
        alpha_family=alpha_family,
        market=market,
        freq=freq,
        features=feature_names,
        partitions=partitions,
        row_count=len(df_factor),
        config_hash=config_hash,
    )
    _atomic_write_text(_metadata_path(store_root, alpha_family, market, freq), json.dumps(metadata, ensure_ascii=False, indent=2))
    if verbose:
        print(
            f"[factor_store] 写入完成: start={metadata['start_date']}, end={metadata['end_date']}, "
            f"partitions={metadata['partition_count']}, rows={metadata['row_count']}"
        )
    return metadata


def _iter_requested_dates(
    start_time: Union[str, pd.Timestamp, None],
    end_time: Union[str, pd.Timestamp, None],
) -> List[str]:
    st = _to_ts(start_time)
    et = _to_ts(end_time)
    if st is None or et is None:
        raise ValueError("start_time and end_time are required")
    if st > et:
        raise ValueError("start_time must be <= end_time")
    dts = pd.date_range(st.normalize(), et.normalize(), freq="D")
    return [d.strftime("%Y-%m-%d") for d in dts]


def read_instruments_list(
    instruments_dir: Union[str, Path],
    market: str,
    start_time: Union[str, pd.Timestamp, None] = None,
    end_time: Union[str, pd.Timestamp, None] = None,
) -> Union[List[str], Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp]]]]:
    """
    读取 instruments 文件，返回股票列表或股票-时间范围字典
    
    Args:
        instruments_dir: instruments 文件所在目录
        market: 市场名称
        start_time: 开始时间
        end_time: 结束时间
    
    Returns:
        返回 {股票代码: [(开始时间1, 结束时间1), (开始时间2, 结束时间2), ...]} 字典
        支持同一股票多条时间范围记录
    """
    instruments_path = Path(instruments_dir).expanduser().resolve() / f"{market}.txt"
    if not instruments_path.exists():
        raise FileNotFoundError(f"instruments 文件不存在: {instruments_path}")
    
    st = _to_ts(start_time)
    et = _to_ts(end_time)
    
    instruments_dict: Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp]]] = {}
    with instruments_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 1:
                inst_code = parts[0]
                inst_start = None
                inst_end = None
                
                if len(parts) >= 3:
                    try:
                        inst_start = pd.Timestamp(parts[1]) if parts[1] else None
                        inst_end = pd.Timestamp(parts[2]) if parts[2] else None
                        
                        if st is not None and inst_end is not None and st > inst_end:
                            continue
                        if et is not None and inst_start is not None and et < inst_start:
                            continue
                    except (ValueError, IndexError):
                        pass
                
                if inst_code not in instruments_dict:
                    instruments_dict[inst_code] = []
                instruments_dict[inst_code].append((inst_start, inst_end))
    
    instruments_list = list(instruments_dict.keys())
    
    print(f"[read_instruments_list] 读取 instruments 列表")
    print(f"  instruments_dir: {instruments_dir}")
    print(f"  market: {market}")
    print(f"  start_time: {start_time}")
    print(f"  end_time: {end_time}")
    print(f"  读取到的股票数: {len(instruments_list)}")
    print(f"  前5个股票: {instruments_list[:5]}")
    
    return instruments_dict


def read_instruments_list_simple(
    instruments_dir: Union[str, Path],
    market: str,
    start_time: Union[str, pd.Timestamp, None] = None,
    end_time: Union[str, pd.Timestamp, None] = None,
) -> List[str]:
    """
    简化版本，只返回股票列表（保持向后兼容）
    """
    instruments_dict = read_instruments_list(instruments_dir, market, start_time, end_time)
    return list(instruments_dict.keys())


def filter_factor_by_market(
    df_factor: pd.DataFrame,
    instruments_dict_or_list: Union[List[str], Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp]]]],
    start_time: Union[str, pd.Timestamp, None] = None,
    end_time: Union[str, pd.Timestamp, None] = None,
) -> pd.DataFrame:
    if isinstance(instruments_dict_or_list, dict):
        instruments_dict = instruments_dict_or_list
        instruments_list = list(instruments_dict.keys())
    else:
        instruments_list = instruments_dict_or_list
        instruments_dict = None
    
    inst_set = set(instruments_list)
    df_filtered = df_factor[df_factor.index.get_level_values("instrument").isin(inst_set)]
    
    if instruments_dict is not None:
        valid_rows_list = []
        dt_idx_all = df_filtered.index.get_level_values("datetime")
        inst_idx_all = df_filtered.index.get_level_values("instrument")
        
        for inst, time_ranges in instruments_dict.items():
            inst_mask = inst_idx_all == inst
            if not inst_mask.any():
                continue
            
            inst_dt = dt_idx_all[inst_mask]
            valid_mask = pd.Series(False, index=inst_dt)
            
            for inst_start, inst_end in time_ranges:
                time_mask = pd.Series(True, index=inst_dt)
                if inst_start is not None:
                    time_mask = time_mask & (inst_dt >= inst_start)
                if inst_end is not None:
                    time_mask = time_mask & (inst_dt <= inst_end)
                valid_mask = valid_mask | time_mask
            
            if valid_mask.any():
                valid_indices = df_filtered.index[inst_mask][valid_mask.values]
                valid_rows_list.append(df_filtered.loc[valid_indices])
        
        if valid_rows_list:
            df_filtered = pd.concat(valid_rows_list)
        else:
            return df_factor.head(0).copy()
    
    if start_time is not None or end_time is not None:
        st = _to_ts(start_time)
        et = _to_ts(end_time)
        
        dt_idx = df_filtered.index.get_level_values("datetime")
        if st is not None:
            df_filtered = df_filtered[dt_idx >= st]
        if et is not None:
            df_filtered = df_filtered[dt_idx <= et]
    
    return df_filtered.sort_index()


def _filtered_store_dir(
    store_root: Union[str, Path],
    alpha_family: str,
    market: str,
    freq: str,
) -> Path:
    return Path(store_root).expanduser().resolve() / alpha_family.lower() / "_filtered" / market.lower() / str(freq).lower()


def _get_cache_key(
    all_metadata: Dict[str, object],
    instruments_path: Path,
) -> str:
    import hashlib
    
    key_parts = []
    key_parts.append(f"schema_version={all_metadata.get('schema_version', '')}")
    key_parts.append(f"start_date={all_metadata.get('start_date', '')}")
    key_parts.append(f"end_date={all_metadata.get('end_date', '')}")
    key_parts.append(f"row_count={all_metadata.get('row_count', '')}")
    
    if instruments_path.exists():
        key_parts.append(f"inst_mtime={instruments_path.stat().st_mtime}")
        key_parts.append(f"inst_size={instruments_path.stat().st_size}")
    
    key_str = "|".join(key_parts)
    return hashlib.md5(key_str.encode("utf-8")).hexdigest()


def _validate_cache(
    cache_dir: Path,
    all_metadata: Dict[str, object],
    instruments_path: Path,
) -> bool:
    cache_key_path = cache_dir / "cache_key.json"
    if not cache_key_path.exists():
        return False
    
    try:
        with cache_key_path.open("r", encoding="utf-8") as f:
            cache_info = json.load(f)
        
        expected_key = _get_cache_key(all_metadata, instruments_path)
        return cache_info.get("cache_key") == expected_key
    except Exception:
        return False


def _get_filter_cache_key(
    all_metadata: Dict[str, object],
    instruments_path: Optional[Path] = None,
    custom_filter_parquet: Optional[Path] = None,
) -> str:
    """
    生成过滤缓存键。
    
    Args:
        all_metadata: 原始因子库元数据
        instruments_path: instruments 文件路径
        custom_filter_parquet: 自定义过滤 parquet 文件路径
    
    Returns:
        str: MD5 哈希值
    
    Example:
        >>> key = _get_filter_cache_key(
        ...     all_metadata={"schema_version": "1.0", ...},
        ...     instruments_path=Path("instruments/csi300.txt"),
        ...     custom_filter_parquet=Path("filter/no_st.parquet"),
        ... )
    """
    import hashlib
    
    key_parts = []
    key_parts.append(f"schema_version={all_metadata.get('schema_version', '')}")
    key_parts.append(f"start_date={all_metadata.get('start_date', '')}")
    key_parts.append(f"end_date={all_metadata.get('end_date', '')}")
    key_parts.append(f"row_count={all_metadata.get('row_count', '')}")
    
    if instruments_path and instruments_path.exists():
        key_parts.append(f"inst_mtime={instruments_path.stat().st_mtime}")
        key_parts.append(f"inst_size={instruments_path.stat().st_size}")
    
    if custom_filter_parquet:
        filter_path = Path(custom_filter_parquet)
        if filter_path.exists():
            key_parts.append(f"filter_mtime={filter_path.stat().st_mtime}")
            key_parts.append(f"filter_size={filter_path.stat().st_size}")
    
    key_str = "|".join(key_parts)
    return hashlib.md5(key_str.encode("utf-8")).hexdigest()


def _get_filtered_cache_dir(
    store_root: Union[str, Path],
    alpha_family: str,
    market: str,
    freq: str,
    custom_filter_parquet: Optional[Union[str, Path]] = None,
) -> Path:
    """
    获取过滤缓存目录。
    
    Args:
        store_root: 因子库根目录
        alpha_family: 因子族名称
        market: 市场代码
        freq: 频率
        custom_filter_parquet: 自定义过滤 parquet 文件路径
    
    Returns:
        Path: 缓存目录路径
    
    Example:
        >>> # 无 parquet 过滤
        >>> dir_path = _get_filtered_cache_dir(
        ...     store_root=".cache/factor_store",
        ...     alpha_family="alpha158_custom",
        ...     market="csi300",
        ...     freq="day",
        ... )
        >>> print(dir_path)
        .cache/factor_store/alpha158_custom/_filtered/csi300/day
        
        >>> # 有 parquet 过滤
        >>> dir_path = _get_filtered_cache_dir(
        ...     store_root=".cache/factor_store",
        ...     alpha_family="alpha158_custom",
        ...     market="csi300",
        ...     freq="day",
        ...     custom_filter_parquet="filter/no_st.parquet",
        ... )
        >>> print(dir_path)
        .cache/factor_store/alpha158_custom/_filtered/csi300_filter_a1b2c3d4/day
    """
    import hashlib
    
    base_dir = Path(store_root).expanduser().resolve() / alpha_family.lower() / "_filtered"
    
    if custom_filter_parquet:
        filter_hash = hashlib.md5(str(custom_filter_parquet).encode()).hexdigest()[:8]
        cache_market = f"{market.lower()}_filter_{filter_hash}"
    else:
        cache_market = market.lower()
    
    return base_dir / cache_market / str(freq).lower()


def _validate_filter_cache(
    cache_dir: Path,
    all_metadata: Dict[str, object],
    instruments_path: Optional[Path] = None,
    custom_filter_parquet: Optional[Path] = None,
    required_start_time: Optional[pd.Timestamp] = None,
    required_end_time: Optional[pd.Timestamp] = None,
) -> bool:
    """
    验证过滤缓存是否有效。
    
    验证条件：
    1. 缓存键匹配
    2. 缓存时间范围覆盖请求时间范围
    
    Args:
        cache_dir: 缓存目录
        all_metadata: 原始因子库元数据
        instruments_path: instruments 文件路径
        custom_filter_parquet: 自定义过滤 parquet 文件路径
        required_start_time: 请求的开始时间
        required_end_time: 请求的结束时间
    
    Returns:
        bool: 缓存是否有效
    
    Example:
        >>> is_valid = _validate_filter_cache(
        ...     cache_dir=Path(".cache/factor_store/alpha158_custom/_filtered/csi300/day"),
        ...     all_metadata={"schema_version": "1.0", ...},
        ...     instruments_path=Path("instruments/csi300.txt"),
        ...     required_start_time=pd.Timestamp("2020-01-01"),
        ...     required_end_time=pd.Timestamp("2021-12-31"),
        ... )
    """
    cache_key_path = cache_dir / "cache_key.json"
    if not cache_key_path.exists():
        return False
    
    try:
        with cache_key_path.open("r", encoding="utf-8") as f:
            cache_info = json.load(f)
        
        expected_key = _get_filter_cache_key(all_metadata, instruments_path, custom_filter_parquet)
        if cache_info.get("cache_key") != expected_key:
            return False
        
        cache_start = cache_info.get("cache_start_date")
        cache_end = cache_info.get("cache_end_date")
        
        if required_start_time is not None and cache_start is not None:
            if pd.Timestamp(required_start_time) < pd.Timestamp(cache_start):
                print(f"[缓存验证] 缓存起始时间 {cache_start} 晚于请求起始时间 {required_start_time}，缓存无效")
                return False
        
        if required_end_time is not None and cache_end is not None:
            if pd.Timestamp(required_end_time) > pd.Timestamp(cache_end):
                print(f"[缓存验证] 缓存结束时间 {cache_end} 早于请求结束时间 {required_end_time}，缓存无效")
                return False
        
        return True
    except Exception as e:
        print(f"[缓存验证] 验证失败: {e}")
        return False


def _filter_by_time_range(
    df: pd.DataFrame,
    start_time: Union[str, pd.Timestamp, None],
    end_time: Union[str, pd.Timestamp, None],
) -> pd.DataFrame:
    """
    按时间范围过滤数据。
    
    Args:
        df: 输入数据
        start_time: 开始时间
        end_time: 结束时间
    
    Returns:
        pd.DataFrame: 过滤后的数据
    """
    if start_time is None and end_time is None:
        return df
    
    dt_idx = df.index.get_level_values("datetime")
    if not isinstance(dt_idx, pd.DatetimeIndex):
        dt_idx = pd.to_datetime(dt_idx)
    
    mask = pd.Series(True, index=df.index)
    if start_time is not None:
        start_time = pd.Timestamp(start_time)
        mask = mask & (dt_idx >= start_time)
    if end_time is not None:
        end_time = pd.Timestamp(end_time)
        mask = mask & (dt_idx <= end_time)
    
    return df.loc[mask]


def _load_factor_slice_from_cache_dir(
    cache_dir: Path,
    start_time: Union[str, pd.Timestamp, None],
    end_time: Union[str, pd.Timestamp, None],
    feature_names: Sequence[str],
) -> pd.DataFrame:
    """
    从缓存目录加载因子数据。
    
    Args:
        cache_dir: 缓存目录
        start_time: 开始时间
        end_time: 结束时间
        feature_names: 特征名列表
    
    Returns:
        pd.DataFrame: 因子数据
    """
    return _load_factor_slice_internal(
        store_root=cache_dir.parent.parent.parent,
        alpha_family=cache_dir.parent.parent.name,
        market=f"_filtered/{cache_dir.parent.name}",
        freq=cache_dir.name,
        feature_names=feature_names,
        start_time=start_time,
        end_time=end_time,
    )


def _save_filtered_cache_v2(
    df: pd.DataFrame,
    cache_dir: Path,
    all_metadata: Dict[str, object],
    instruments_path: Optional[Path] = None,
    custom_filter_path: Optional[Path] = None,
) -> None:
    """
    保存过滤后的因子数据到缓存。
    
    Args:
        df: 过滤后的因子数据
        cache_dir: 缓存目录
        all_metadata: 原始因子库元数据
        instruments_path: instruments 文件路径
        custom_filter_path: 自定义过滤 parquet 文件路径
    """
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    build_factor_store(
        df_factor=df,
        store_root=cache_dir.parent.parent.parent,
        alpha_family=cache_dir.parent.parent.name,
        market=f"_filtered/{cache_dir.parent.name}",
        freq=cache_dir.name,
        overwrite=True,
        verbose=False,
    )
    
    dt_idx = df.index.get_level_values("datetime")
    cache_start = dt_idx.min().strftime("%Y-%m-%d") if len(dt_idx) > 0 else None
    cache_end = dt_idx.max().strftime("%Y-%m-%d") if len(dt_idx) > 0 else None
    
    cache_key = _get_filter_cache_key(all_metadata, instruments_path, custom_filter_path)
    cache_key_path = cache_dir / "cache_key.json"
    cache_info = {
        "cache_key": cache_key,
        "cache_start_date": cache_start,
        "cache_end_date": cache_end,
        "row_count": len(df),
    }
    _atomic_write_text(cache_key_path, json.dumps(cache_info, ensure_ascii=False, indent=2))
    print(f"[缓存保存] 缓存时间范围: {cache_start} ~ {cache_end}, 数据行数: {len(df)}")


def load_factor_slice(
    store_root: Union[str, Path],
    alpha_family: str,
    market: str,
    freq: str,
    feature_names: Sequence[str],
    start_time: Union[str, pd.Timestamp, None] = None,
    end_time: Union[str, pd.Timestamp, None] = None,
    filter_market: Optional[str] = None,
    filter_parquet: Union[str, Path, None] = None,
    filter_instruments_dir: Union[str, Path, None] = None,
    filter_parquet_date_col: str = "date",
    filter_parquet_inst_col: str = "ts_code",
    cache_filtered: bool = True,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    加载因子数据切片，支持缓存和多种过滤方式。
    
    数据源：从 market 指定的缓存市场加载原始数据
    过滤：应用 filter_market 和 filter_parquet 过滤
    缓存：如果 cache_filtered=True，缓存过滤后的结果
    
    过滤优先级：
    1. instruments txt 文件过滤（filter_market + filter_instruments_dir）
    2. parquet 文件过滤（filter_parquet）
    
    缓存策略：
    - 如果 force_refresh=True，强制重新计算，不使用缓存
    - 否则，优先检查 _filtered 目录中的缓存
    - 缓存有效则直接加载，否则重新计算并保存缓存
    
    Args:
        store_root: 因子库根目录
        alpha_family: 因子族名称，如 "alpha158_custom", "alpha101", "alpha191"
        market: 缓存市场/数据源，如 "csi300", "all"
        freq: 频率，如 "day"
        feature_names: 需要加载的特征名列表
        start_time: 开始时间
        end_time: 结束时间
        filter_market: 过滤条件（instruments txt 文件），如 "csi300_no_suspend_min_90_days_in_csi300"
        filter_parquet: 过滤条件（parquet 文件）
        filter_instruments_dir: instruments 文件目录
        filter_parquet_date_col: parquet 中的日期列名
        filter_parquet_inst_col: parquet 中的股票代码列名
        cache_filtered: 是否缓存过滤结果
        force_refresh: 是否强制刷新缓存
    
    Returns:
        pd.DataFrame: 因子数据，索引为 MultiIndex(datetime, instrument)
    
    Example:
        >>> # 示例1：加载 csi300 缓存因子，无过滤
        >>> df = load_factor_slice(
        ...     store_root=".cache/factor_store",
        ...     alpha_family="alpha158_custom",
        ...     market="csi300",
        ...     freq="day",
        ...     feature_names=["FEATURE0", "FEATURE1"],
        ...     start_time="2020-01-01",
        ...     end_time="2021-12-31",
        ... )
        
        >>> # 示例2：从 all 缓存加载，应用 filter_market 过滤
        >>> df = load_factor_slice(
        ...     store_root=".cache/factor_store",
        ...     alpha_family="alpha158_custom",
        ...     market="all",
        ...     freq="day",
        ...     feature_names=["FEATURE0", "FEATURE1"],
        ...     start_time="2020-01-01",
        ...     end_time="2021-12-31",
        ...     filter_market="csi300_no_suspend_min_90_days_in_csi300",
        ...     filter_instruments_dir="qlib_data_bin/instruments",
        ... )
        
        >>> # 示例3：加载因子，应用 parquet 过滤
        >>> df = load_factor_slice(
        ...     store_root=".cache/factor_store",
        ...     alpha_family="alpha158_custom",
        ...     market="all",
        ...     freq="day",
        ...     feature_names=["FEATURE0", "FEATURE1"],
        ...     start_time="2020-01-01",
        ...     end_time="2021-12-31",
        ...     filter_parquet="filter/no_st_no_suspend.parquet",
        ... )
    """
    print(f"\n{'='*60}")
    print(f"[load_factor_slice] 开始加载因子数据")
    print(f"  alpha_family: {alpha_family}")
    print(f"  market (cache_market): {market}")
    print(f"  filter_market: {filter_market}")
    print(f"  freq: {freq}")
    print(f"  start_time: {start_time}")
    print(f"  end_time: {end_time}")
    print(f"  filter_instruments_dir: {filter_instruments_dir}")
    print(f"  cache_filtered: {cache_filtered}")
    print(f"  filter_parquet: {filter_parquet}")
    print(f"  force_refresh: {force_refresh}")
    print(f"{'='*60}\n")
    
    need_instruments_filter = (
        filter_market is not None 
        and filter_instruments_dir is not None
    )
    need_parquet_filter = filter_parquet is not None
    need_any_filter = need_instruments_filter or need_parquet_filter
    
    instruments_path = None
    if need_instruments_filter:
        instruments_path = Path(filter_instruments_dir).expanduser().resolve() / f"{filter_market}.txt"
    
    custom_filter_path = None
    if need_parquet_filter:
        custom_filter_path = Path(filter_parquet).expanduser().resolve()
    
    if not need_any_filter:
        print(f"[load_factor_slice] 直接加载原始因子库: {market}")
        return _load_factor_slice_internal(
            store_root=store_root,
            alpha_family=alpha_family,
            market=market,
            freq=freq,
            feature_names=feature_names,
            start_time=start_time,
            end_time=end_time,
        )
    
    cache_dir = _get_filtered_cache_dir(store_root, alpha_family, filter_market or market, freq, filter_parquet)
    
    if cache_filtered and not force_refresh:
        try:
            source_metadata = read_factor_store_metadata(store_root, alpha_family, market, freq)
            if _validate_filter_cache(
                cache_dir, 
                source_metadata, 
                instruments_path, 
                custom_filter_path,
                required_start_time=start_time,
                required_end_time=end_time,
            ):
                print(f"[load_factor_slice] 使用缓存: {cache_dir}")
                return _load_factor_slice_from_cache_dir(
                    cache_dir=cache_dir,
                    start_time=start_time,
                    end_time=end_time,
                    feature_names=feature_names,
                )
        except Exception as e:
            print(f"[load_factor_slice] 缓存检查失败: {e}")
    
    source_dir = _store_dir(store_root, alpha_family, market, freq)
    if not source_dir.exists():
        raise FileNotFoundError(f"因子库不存在: {source_dir}")
    
    print(f"[load_factor_slice] 从 {market} 市场加载并过滤")
    df = _load_factor_slice_internal(
        store_root=store_root,
        alpha_family=alpha_family,
        market=market,
        freq=freq,
        feature_names=feature_names,
        start_time=None,
        end_time=None,
    )
    
    print(f"[load_factor_slice] 过滤前数据形状: {df.shape}")
    
    if need_instruments_filter:
        print(f"[load_factor_slice] 执行 instruments 过滤: {instruments_path}")
        instruments_dict = read_instruments_list(filter_instruments_dir, filter_market, None, None)
        df = filter_factor_by_market(df, instruments_dict, None, None)
        print(f"[load_factor_slice] instruments 过滤后数据形状: {df.shape}")
    
    if need_parquet_filter:
        print(f"[load_factor_slice] 执行 parquet 过滤: {custom_filter_path}")
        df = filter_df_by_parquet(df, custom_filter_path, filter_parquet_date_col, filter_parquet_inst_col)
        print(f"[load_factor_slice] parquet 过滤后数据形状: {df.shape}")
    
    if cache_filtered:
        try:
            source_metadata = read_factor_store_metadata(store_root, alpha_family, market, freq)
            _save_filtered_cache_v2(
                df=df,
                cache_dir=cache_dir,
                all_metadata=source_metadata,
                instruments_path=instruments_path,
                custom_filter_path=custom_filter_path,
            )
            print(f"[load_factor_slice] 已保存缓存: {cache_dir}")
        except Exception as e:
            print(f"[load_factor_slice] 保存缓存失败: {e}")
    
    if start_time is not None or end_time is not None:
        df = _filter_by_time_range(df, start_time, end_time)
        print(f"[load_factor_slice] 时间切片后数据形状: {df.shape}")
    
    return df


def _get_memory_cache_key(alpha_family: str, market: str, freq: str) -> str:
    """生成内存缓存键"""
    return f"{alpha_family}|{market}|{freq}"


def _check_cache_covers_range(
    cache_key: str,
    start_time: Optional[pd.Timestamp],
    end_time: Optional[pd.Timestamp],
) -> bool:
    """检查缓存是否覆盖请求的时间范围"""
    if cache_key not in _cache_time_range:
        return False
    
    cache_start, cache_end = _cache_time_range[cache_key]
    
    if start_time is not None and cache_start is not None:
        if start_time < cache_start:
            return False
    
    if end_time is not None and cache_end is not None:
        if end_time > cache_end:
            return False
    
    return True


def _load_factor_slice_internal(
    store_root: Union[str, Path],
    alpha_family: str,
    market: str,
    freq: str,
    feature_names: Sequence[str],
    start_time: Union[str, pd.Timestamp, None],
    end_time: Union[str, pd.Timestamp, None],
) -> pd.DataFrame:
    import time
    t0 = time.time()
    
    cache_key = _get_memory_cache_key(alpha_family, market, freq)
    st = _to_ts(start_time)
    et = _to_ts(end_time)
    
    with _cache_lock:
        if cache_key in _factor_memory_cache:
            if _check_cache_covers_range(cache_key, st, et):
                _cache_stats["hits"] += 1
                cached_df = _factor_memory_cache[cache_key]
                cache_start, cache_end = _cache_time_range[cache_key]
                print(f'[内存缓存] 命中! key={cache_key}, 缓存范围={cache_start}~{cache_end}, 缓存大小={cached_df.shape}')
                
                result = cached_df.copy()
                if st is not None:
                    result = result[result.index.get_level_values("datetime") >= st]
                if et is not None:
                    result = result[result.index.get_level_values("datetime") <= et]
                
                print(f'load_factor_slice: {result.shape} (从内存缓存切片)')
                return result
            else:
                cache_start, cache_end = _cache_time_range.get(cache_key, (None, None))
                print(f'[内存缓存] 缓存范围不足! key={cache_key}')
                print(f'  缓存范围: {cache_start}~{cache_end}')
                print(f'  请求范围: {st}~{et}')
                print(f'  重新加载全量数据...')
                _cache_stats["misses"] += 1
        else:
            _cache_stats["misses"] += 1
    
    target_dir = _store_dir(store_root, alpha_family, market, freq)
    available_dates = _list_partitions(target_dir)
    
    if not available_dates:
        raise FileNotFoundError(f"因子库无可用分区: {target_dir}")
    
    print(f'alpha_family: {alpha_family}, market: {market}, freq: {freq}, 加载全量数据: {available_dates[0]}~{available_dates[-1]}')
    
    t1 = time.time()
    
    file_list = []
    for dt_str in available_dates:
        part_dir = target_dir / f"datetime={dt_str}"
        files = sorted(part_dir.glob("*.parquet"))
        if files:
            file_list.append(files[0])
    
    if not file_list:
        return pd.DataFrame(columns=list(feature_names), index=pd.MultiIndex.from_arrays([[], []], names=["datetime", "instrument"]))
    
    chunks = _read_parquet_files_parallel(file_list, list(feature_names))
    
    t2 = time.time()
    print(f'  [性能] 并行读取 {len(chunks)} 个 parquet 文件耗时: {t2-t1:.2f}秒')
    
    out = pd.concat(chunks, axis=0, ignore_index=True)
    t3 = time.time()
    print(f'  [性能] concat 合并耗时: {t3-t2:.2f}秒')
    
    out["datetime"] = pd.to_datetime(out["datetime"])
    out = out.set_index(["datetime", "instrument"]).sort_index()
    out = out.loc[:, list(feature_names)]
    
    t4 = time.time()
    print(f'  [性能] 设置索引耗时: {t4-t3:.2f}秒')
    print(f'  [性能] 总耗时: {t4-t0:.2f}秒')
    
    cache_start = out.index.get_level_values("datetime").min()
    cache_end = out.index.get_level_values("datetime").max()
    
    with _cache_lock:
        _factor_memory_cache[cache_key] = out.copy()
        _cache_time_range[cache_key] = (cache_start, cache_end)
        print(f'[内存缓存] 已缓存 key={cache_key}, 范围={cache_start}~{cache_end}, 大小={out.shape}')
    
    result = out.copy()
    if st is not None:
        result = result[result.index.get_level_values("datetime") >= st]
    if et is not None:
        result = result[result.index.get_level_values("datetime") <= et]
    
    print(f'load_factor_slice: {result.shape}')
    return result


def _read_single_parquet(file_path: Path, columns: List[str]) -> pd.DataFrame:
    """读取单个 parquet 文件"""
    return pd.read_parquet(file_path, columns=columns)


def _read_parquet_files_parallel(file_list: List[Path], columns: List[str], n_jobs: int = -1) -> List[pd.DataFrame]:
    """
    并行读取多个 parquet 文件
    
    Args:
        file_list: 文件路径列表
        columns: 需要读取的列名
        n_jobs: 并行进程数，-1 表示使用所有 CPU 核心
    
    Returns:
        DataFrame 列表
    """
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed
    
    if n_jobs == -1:
        n_jobs = os.cpu_count() or 4
    
    n_jobs = min(n_jobs, len(file_list))
    
    if n_jobs <= 1 or len(file_list) <= 10:
        return [pd.read_parquet(f, columns=["datetime", "instrument", *columns]) for f in file_list]
    
    full_columns = ["datetime", "instrument", *columns]
    
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = {executor.submit(_read_single_parquet, f, full_columns): f for f in file_list}
        results = []
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                file_path = futures[future]
                print(f"  [警告] 读取文件失败: {file_path}, 错误: {e}")
    
    return results


def _load_factor_slice_from_cache(
    store_root: Union[str, Path],
    alpha_family: str,
    market: str,
    freq: str,
    feature_names: Sequence[str],
    start_time: Union[str, pd.Timestamp, None],
    end_time: Union[str, pd.Timestamp, None],
) -> pd.DataFrame:
    print(f'Loading from cache: {market}')
    return _load_factor_slice_internal(
        store_root=store_root,
        alpha_family=alpha_family,
        market=f"_filtered/{market}",
        freq=freq,
        feature_names=feature_names,
        start_time=start_time,
        end_time=end_time,
    )


def _save_filtered_cache(
    df_filtered: pd.DataFrame,
    store_root: Union[str, Path],
    alpha_family: str,
    market: str,
    freq: str,
    all_metadata: Dict[str, object],
    instruments_path: Path,
) -> None:
    cache_dir = _filtered_store_dir(store_root, alpha_family, market, freq)
    
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    
    build_factor_store(
        df_factor=df_filtered,
        store_root=store_root,
        alpha_family=alpha_family,
        market=f"_filtered/{market}",
        freq=freq,
        overwrite=True,
        verbose=False,
    )
    
    cache_key = _get_cache_key(all_metadata, instruments_path)
    cache_key_path = cache_dir / "cache_key.json"
    _atomic_write_text(cache_key_path, json.dumps({"cache_key": cache_key}, ensure_ascii=False, indent=2))
    
    print(f'Cache saved for: {market}')


_custom_filter_parquet_cache: Dict[str, Tuple[pd.DataFrame, set, tuple]] = {}


def _load_and_parse_filter_parquet(
    parquet_path: Union[str, Path],
    date_col: str,
    inst_col: str,
) -> Tuple[pd.DataFrame, set, tuple]:
    path = Path(parquet_path).expanduser().resolve()
    cache_key = str(path)
    cached = _custom_filter_parquet_cache.get(cache_key)
    if cached is not None:
        df_filter, valid_set, col_info = cached
        if path.stat().st_mtime == getattr(_load_and_parse_filter_parquet, f"_mtime_{id(path)}", None):
            return df_filter, valid_set, col_info

    df_filter = pd.read_parquet(path)
    required_cols = {date_col, inst_col}
    missing = required_cols - set(df_filter.columns)
    if missing:
        raise ValueError(
            f"parquet文件缺少必要的列: {missing}. "
            f"现有列: {df_filter.columns.tolist()}. "
            f"请通过 custom_filter_date_col 和 custom_filter_inst_col 参数指定正确的列名."
        )

    date_series = pd.to_datetime(df_filter[date_col])
    inst_series = df_filter[inst_col].astype(str)
    valid_set = set(zip(date_series.values, inst_series.values))
    col_info = (date_col, inst_col)

    setattr(_load_and_parse_filter_parquet, f"_mtime_{id(path)}", path.stat().st_mtime)
    _custom_filter_parquet_cache[cache_key] = (df_filter, valid_set, col_info)

    print(f"[filter_df_by_parquet] 加载过滤parquet: {path}")
    print(f"  日期列: {date_col}, 股票列: {inst_col}")
    print(f"  过滤记录数: {len(valid_set)}")
    print(f"  日期范围: {date_series.min()} ~ {date_series.max()}")
    print(f"  唯一股票数: {inst_series.nunique()}")

    return df_filter, valid_set, col_info


def filter_df_by_parquet(
    df: pd.DataFrame,
    parquet_path: Union[str, Path],
    date_col: str = "date",
    inst_col: str = "ts_code",
) -> pd.DataFrame:
    """根据parquet文件中的(datetime, instrument)组合过滤DataFrame，作为数据链路最后环节的自定义过滤.

    Args:
        df: 待过滤的DataFrame，索引必须是 MultiIndex(datetime, instrument)
        parquet_path: 过滤条件parquet文件路径，包含 date_col 和 inst_col 两列
        date_col: parquet中的日期列名，默认 "date"
        inst_col: parquet中的股票代码列名，默认 "ts_code"

    Returns:
        过滤后的DataFrame，仅保留parquet中存在的(datetime, instrument)组合
    """
    _, valid_set, (actual_date_col, actual_inst_col) = _load_and_parse_filter_parquet(
        parquet_path, date_col, inst_col
    )

    original_shape = df.shape

    dt_idx = df.index.get_level_values("datetime")
    if not isinstance(dt_idx, pd.DatetimeIndex):
        dt_idx = pd.to_datetime(dt_idx)
    inst_idx = df.index.get_level_values("instrument").astype(str)

    valid_index = pd.MultiIndex.from_tuples(valid_set, names=['datetime', 'instrument'])
    df_index = pd.MultiIndex.from_arrays([dt_idx.values, inst_idx.values], names=['datetime', 'instrument'])
    mask = df_index.isin(valid_index)
    df_filtered = df.loc[mask]

    print(
        f"[filter_df_by_parquet] 过滤结果: {original_shape} -> {df_filtered.shape}, "
        f"移除 {original_shape[0] - df_filtered.shape[0]} 行"
    )

    return df_filtered


def _get_label_cache_dir(
    store_root: Union[str, Path],
    alpha_family: str,
    filter_market: Optional[str],
    filter_parquet: Optional[Union[str, Path]],
    label_names: Sequence[str],
    freq: str,
) -> Path:
    """
    获取 label 缓存目录路径。
    
    目录结构: {store_root}/{alpha_family}/_label/{filter_condition}/{label_hash}/{freq}/
    """
    import hashlib
    
    filter_parts = []
    if filter_market:
        filter_parts.append(filter_market)
    if filter_parquet:
        filter_parts.append(Path(filter_parquet).stem)
    
    filter_condition = "_".join(filter_parts) if filter_parts else "no_filter"
    
    label_hash = hashlib.md5("|".join(sorted(label_names)).encode()).hexdigest()[:8]
    
    return Path(store_root).expanduser().resolve() / alpha_family.lower() / "_label" / filter_condition / label_hash / str(freq).lower()


def _get_label_cache_info_path(
    store_root: Union[str, Path],
    alpha_family: str,
    filter_market: Optional[str],
    filter_parquet: Optional[Union[str, Path]],
    label_names: Sequence[str],
    freq: str,
) -> Path:
    """获取 label 缓存信息文件路径"""
    cache_dir = _get_label_cache_dir(store_root, alpha_family, filter_market, filter_parquet, label_names, freq)
    return cache_dir / "cache_info.json"


def load_label_cache(
    store_root: Union[str, Path],
    alpha_family: str,
    filter_market: Optional[str],
    filter_parquet: Optional[Union[str, Path]],
    label_names: Sequence[str],
    freq: str,
    start_time: Union[str, pd.Timestamp, None],
    end_time: Union[str, pd.Timestamp, None],
) -> Optional[pd.DataFrame]:
    """
    加载 label 缓存。
    
    缓存存储全量数据，加载时按时间切片返回。
    
    Args:
        store_root: 缓存根目录
        alpha_family: 因子族名称（如 "alpha158_custom"）
        filter_market: 过滤条件（instruments txt）
        filter_parquet: 过滤条件（parquet 文件）
        label_names: label 名称列表
        freq: 频率
        start_time: 开始时间（用于切片）
        end_time: 结束时间（用于切片）
    
    Returns:
        缓存的 DataFrame（已按时间切片），如果缓存不存在或无效则返回 None
    """
    cache_dir = _get_label_cache_dir(store_root, alpha_family, filter_market, filter_parquet, label_names, freq)
    cache_info_path = _get_label_cache_info_path(store_root, alpha_family, filter_market, filter_parquet, label_names, freq)
    
    if not cache_dir.exists() or not cache_info_path.exists():
        print(f"[load_label_cache] 缓存不存在: {cache_dir}")
        return None
    
    try:
        with cache_info_path.open("r", encoding="utf-8") as f:
            cache_info = json.load(f)
        
        cache_labels = set(cache_info.get("label_names", []))
        
        if set(label_names) != cache_labels:
            print(f"[load_label_cache] label 名称不匹配: 缓存 {cache_labels}, 请求 {set(label_names)}")
            return None
        
        cache_start = cache_info.get("cache_start_date")
        cache_end = cache_info.get("cache_end_date")
        
        print(f"[load_label_cache] 使用缓存: {cache_dir}")
        print(f"  filter_market: {filter_market}, filter_parquet: {filter_parquet}")
        print(f"  label_names: {label_names}, freq: {freq}")
        print(f"  缓存时间范围: {cache_start} ~ {cache_end}")
        
        df = _load_factor_slice_internal(
            store_root=cache_dir.parent.parent.parent.parent,
            alpha_family="_label",
            market=f"{cache_dir.parent.parent.name}/{cache_dir.parent.name}",
            freq=freq,
            feature_names=label_names,
            start_time=None,
            end_time=None,
        )
        
        st = _to_ts(start_time)
        et = _to_ts(end_time)
        
        if st is not None:
            df = df[df.index.get_level_values("datetime") >= st]
        if et is not None:
            df = df[df.index.get_level_values("datetime") <= et]
        
        print(f"[load_label_cache] 时间切片后: {df.shape}")
        
        return df
    except Exception as e:
        print(f"[load_label_cache] 加载缓存失败: {e}")
        return None


def save_label_cache(
    df: pd.DataFrame,
    store_root: Union[str, Path],
    alpha_family: str,
    filter_market: Optional[str],
    filter_parquet: Optional[Union[str, Path]],
    label_names: Sequence[str],
    freq: str,
) -> None:
    """
    保存 label 缓存。
    
    Args:
        df: label 数据
        store_root: 缓存根目录
        alpha_family: 因子族名称（如 "alpha158_custom"）
        filter_market: 过滤条件（instruments txt）
        filter_parquet: 过滤条件（parquet 文件）
        label_names: label 名称列表
        freq: 频率
    """
    cache_dir = _get_label_cache_dir(store_root, alpha_family, filter_market, filter_parquet, label_names, freq)
    cache_info_path = _get_label_cache_info_path(store_root, alpha_family, filter_market, filter_parquet, label_names, freq)
    
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    build_factor_store(
        df_factor=df,
        store_root=cache_dir.parent.parent.parent.parent,
        alpha_family="_label",
        market=f"{cache_dir.parent.parent.name}/{cache_dir.parent.name}",
        freq=freq,
        overwrite=True,
        verbose=False,
    )
    
    dt_idx = df.index.get_level_values("datetime")
    cache_start = dt_idx.min().strftime("%Y-%m-%d") if len(dt_idx) > 0 else None
    cache_end = dt_idx.max().strftime("%Y-%m-%d") if len(dt_idx) > 0 else None
    
    cache_info = {
        "cache_start_date": cache_start,
        "cache_end_date": cache_end,
        "row_count": len(df),
        "label_names": list(label_names),
        "filter_market": filter_market,
        "filter_parquet": str(filter_parquet) if filter_parquet else None,
        "freq": freq,
    }
    
    _atomic_write_text(cache_info_path, json.dumps(cache_info, ensure_ascii=False, indent=2))
    
    print(f"[save_label_cache] 已保存缓存: {cache_dir}")
    print(f"  时间范围: {cache_start} ~ {cache_end}, 数据行数: {len(df)}")
