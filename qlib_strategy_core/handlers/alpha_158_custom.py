from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from qlib.contrib.data.handler import _DEFAULT_LEARN_PROCESSORS, check_transform_proc
from qlib.contrib.data.loader import Alpha158DL
from qlib.data import D
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset.loader import DLWParser

from qlib_strategy_core.alpha_factor_store import (
    load_factor_slice,
    filter_df_by_parquet,
    read_instruments_list,
    filter_factor_by_market,
    load_label_cache,
    save_label_cache,
)


class Alpha158CustomDataLoader(DLWParser):
    """
    Alpha158 自定义因子数据加载器。
    
    支持从因子库缓存加载因子数据，或从 QLib 原始数据计算。
    过滤功能独立于缓存模式，无论是否使用缓存都可以应用过滤。
    
    参数说明：
    - instruments: 股票池/缓存市场。use_cache=True 时作为缓存市场，use_cache=False 时作为 QLib 股票池
    - use_cache: 是否使用因子库缓存。True 时从缓存加载，False 时从 QLib 计算
    - filter_market: 过滤条件（instruments txt 文件），如 "csi300_no_suspend_min_90_days_in_csi300"
    - filter_parquet: 过滤条件（parquet 文件）
    
    过滤优先级：
    1. 先应用 filter_market 过滤（instruments txt）
    2. 再应用 filter_parquet 过滤
    
    Example:
        >>> # 示例1：使用缓存 + 过滤
        >>> handler = Alpha158Custom(
        ...     instruments="csi300",  # 从 csi300 缓存加载 feature
        ...     start_time="2020-01-01",
        ...     end_time="2021-12-31",
        ...     use_cache=True,
        ...     cache_root=".cache/factor_store",
        ...     filter_market="csi300_no_suspend_min_90_days_in_csi300",
        ...     filter_instruments_dir="qlib_data_bin/instruments",
        ... )
        
        >>> # 示例2：不使用缓存 + 过滤
        >>> handler = Alpha158Custom(
        ...     instruments="csi300",  # 从 QLib 加载 csi300 数据
        ...     start_time="2020-01-01",
        ...     end_time="2021-12-31",
        ...     use_cache=False,
        ...     filter_market="csi300_no_suspend_min_90_days_in_csi300",
        ...     filter_instruments_dir="qlib_data_bin/instruments",
        ... )
    """
    
    def __init__(
        self,
        config,
        filter_pipe=None,
        freq: Union[str, dict] = "day",
        inst_processors=None,
        use_cache: bool = True,
        cache_root: Union[str, Path, None] = None,
        cache_filtered: bool = True,
        cache_label: bool = True,
        filter_market: Optional[str] = None,
        filter_parquet: Union[str, Path, None] = None,
        filter_instruments_dir: Union[str, Path, None] = None,
        filter_parquet_date_col: str = "date",
        filter_parquet_inst_col: str = "ts_code",
        force_refresh: bool = False,
        n_jobs: int = 1,
        parallel_backend: str = "threads",
        share_data: str = "none",
        show_progress: bool = False,
        progress_every: int = 10,
    ):
        self.filter_pipe = filter_pipe
        self.freq = freq
        self.inst_processors = inst_processors if inst_processors is not None else {}
        self.use_cache = use_cache
        self.cache_root = cache_root
        self.cache_filtered = cache_filtered
        self.cache_label = cache_label
        self.filter_market = filter_market
        self.filter_parquet = filter_parquet
        self.filter_instruments_dir = filter_instruments_dir
        self.filter_parquet_date_col = filter_parquet_date_col
        self.filter_parquet_inst_col = filter_parquet_inst_col
        self.force_refresh = force_refresh
        assert isinstance(self.inst_processors, (dict, list))
        super().__init__(config)

    def load_group_df(
        self,
        instruments,
        exprs: list,
        names: list,
        start_time: Union[str, pd.Timestamp] = None,
        end_time: Union[str, pd.Timestamp] = None,
        gp_name: str = None,
    ) -> pd.DataFrame:
        raw_instruments = instruments
        if instruments is None:
            warnings.warn("`instruments` is not set, will load all stocks")
            instruments = "all"
        if isinstance(instruments, str):
            instruments = D.instruments(instruments, filter_pipe=self.filter_pipe)
        elif self.filter_pipe is not None:
            warnings.warn("`filter_pipe` is not None, but it will not be used with `instruments` as list")

        freq = self.freq[gp_name] if isinstance(self.freq, dict) else self.freq
        inst_processors = (
            self.inst_processors if isinstance(self.inst_processors, list) else self.inst_processors.get(gp_name, [])
        )
        
        print(f"[DEBUG] load_group_df: gp_name={gp_name}, use_cache={self.use_cache}")
        
        if gp_name == "feature" and self.use_cache:
            print(f"[DEBUG] 进入缓存分支，调用 _load_feature_from_cache")
            df = self._load_feature_from_cache(raw_instruments, names, start_time, end_time, freq)
            print(f"[DEBUG] 缓存加载完成，df.shape={df.shape}，跳过过滤")
        elif gp_name == "label" and self.use_cache and self.cache_label:
            print(f"[DEBUG] 进入 label 缓存分支")
            df = self._load_label_from_cache(names, start_time, end_time, freq)
            if df is None:
                print(f"[DEBUG] label 缓存未命中，加载全量数据并缓存")
                df_full = self._load_from_qlib(instruments, exprs, names, None, None, freq, inst_processors)
                print(f"[DEBUG] QLib加载全量数据完成，df_full.shape={df_full.shape}，执行过滤")
                df_full = self._apply_filters(df_full, None, None, gp_name)
                print(f"[DEBUG] 全量数据过滤完成，df_full.shape={df_full.shape}，保存缓存")
                self._save_label_cache(df_full, names, freq)
                st = pd.Timestamp(start_time) if start_time else None
                et = pd.Timestamp(end_time) if end_time else None
                df = df_full.copy()
                if st is not None:
                    df = df[df.index.get_level_values("datetime") >= st]
                if et is not None:
                    df = df[df.index.get_level_values("datetime") <= et]
                print(f"[DEBUG] 时间切片后，df.shape={df.shape}")
            else:
                print(f"[DEBUG] label 缓存命中，df.shape={df.shape}，跳过过滤")
        else:
            print(f"[DEBUG] 进入非缓存分支，调用 _load_from_qlib")
            df = self._load_from_qlib(instruments, exprs, names, start_time, end_time, freq, inst_processors)
            print(f"[DEBUG] QLib加载完成，df.shape={df.shape}，执行过滤")
            df = self._apply_filters(df, start_time, end_time, gp_name)
        
        return df

    def _load_feature_from_cache(self, market, feature_names, start_time, end_time, freq):
        if self.cache_root is None:
            raise ValueError("use_cache=True 时必须提供 cache_root")
        
        if not isinstance(market, str):
            raise ValueError(f"use_cache=True 时 instruments 必须是字符串（缓存市场名），当前类型: {type(market)}")
        
        return load_factor_slice(
            store_root=self.cache_root,
            alpha_family="alpha158_custom",
            market=market,
            freq=freq,
            feature_names=feature_names,
            start_time=start_time,
            end_time=end_time,
            filter_market=self.filter_market,
            filter_parquet=self.filter_parquet,
            filter_instruments_dir=self.filter_instruments_dir,
            filter_parquet_date_col=self.filter_parquet_date_col,
            filter_parquet_inst_col=self.filter_parquet_inst_col,
            cache_filtered=self.cache_filtered,
            force_refresh=self.force_refresh,
        )

    def _load_label_from_cache(self, label_names, start_time, end_time, freq):
        """从缓存加载 label 数据"""
        if self.cache_root is None:
            return None
        
        return load_label_cache(
            store_root=self.cache_root,
            alpha_family="alpha158_custom",
            filter_market=self.filter_market,
            filter_parquet=self.filter_parquet,
            label_names=label_names,
            freq=freq,
            start_time=start_time,
            end_time=end_time,
        )

    def _save_label_cache(self, df, label_names, freq):
        """保存 label 数据到缓存"""
        if self.cache_root is None:
            return
        
        save_label_cache(
            df=df,
            store_root=self.cache_root,
            alpha_family="alpha158_custom",
            filter_market=self.filter_market,
            filter_parquet=self.filter_parquet,
            label_names=label_names,
            freq=freq,
        )

    def _load_from_qlib(self, instruments, exprs, names, start_time, end_time, freq, inst_processors):
        df = D.features(instruments, exprs, start_time, end_time, freq=freq, inst_processors=inst_processors)
        df.columns = names
        df = df.swaplevel().sort_index()
        return df

    def _apply_filters(self, df, start_time, end_time, gp_name):
        if self.filter_market and self.filter_instruments_dir:
            instruments_dict = read_instruments_list(
                self.filter_instruments_dir,
                self.filter_market,
                start_time,
                end_time,
            )
            df = filter_factor_by_market(df, instruments_dict, start_time, end_time)
            print(f"[load_group_df] {gp_name} 应用 filter_market 过滤: {self.filter_market}, 过滤后形状: {df.shape}")
        
        if self.filter_parquet:
            df = filter_df_by_parquet(
                df,
                parquet_path=self.filter_parquet,
                date_col=self.filter_parquet_date_col,
                inst_col=self.filter_parquet_inst_col,
            )
            print(f"[load_group_df] {gp_name} 应用 filter_parquet 过滤, 过滤后形状: {df.shape}")
        
        return df


class Alpha158Custom(DataHandlerLP):
    """
    Alpha158 自定义因子处理器。
    
    支持从因子库缓存加载因子数据，或从 QLib 原始数据计算。
    过滤功能独立于缓存模式。
    
    参数说明：
    - instruments: 股票池/缓存市场。use_cache=True 时作为缓存市场，use_cache=False 时作为 QLib 股票池
    - use_cache: 是否使用因子库缓存
    - cache_root: 因子库根目录
    - cache_filtered: 是否缓存过滤后的 feature 数据
    - cache_label: 是否缓存 label 数据（默认 True）
    - filter_market: 过滤条件（instruments txt 文件）
    - filter_parquet: 过滤条件（parquet 文件）
    - filter_instruments_dir: instruments 文件目录
    
    Example:
        >>> # 示例1：使用缓存 + 过滤
        >>> handler = Alpha158Custom(
        ...     instruments="csi300",  # 从 csi300 缓存加载 feature
        ...     start_time="2020-01-01",
        ...     end_time="2021-12-31",
        ...     use_cache=True,
        ...     cache_root=".cache/factor_store",
        ...     filter_market="csi300_no_suspend_min_90_days_in_csi300",
        ...     filter_instruments_dir="qlib_data_bin/instruments",
        ... )
        
        >>> # 示例2：不使用缓存 + 过滤
        >>> handler = Alpha158Custom(
        ...     instruments="csi300",  # 从 QLib 加载 csi300 数据
        ...     start_time="2020-01-01",
        ...     end_time="2021-12-31",
        ...     use_cache=False,
        ...     filter_market="csi300_no_suspend_min_90_days_in_csi300",
        ...     filter_instruments_dir="qlib_data_bin/instruments",
        ... )
    """
    
    def __init__(
        self,
        instruments="csi500",
        start_time=None,
        end_time=None,
        freq="day",
        infer_processors=None,
        learn_processors=_DEFAULT_LEARN_PROCESSORS,
        fit_start_time=None,
        fit_end_time=None,
        process_type=DataHandlerLP.PTYPE_A,
        filter_pipe=None,
        inst_processors=None,
        use_cache: bool = True,
        cache_root: Union[str, Path, None] = None,
        cache_filtered: bool = True,
        filter_market: Optional[str] = None,
        filter_parquet: Union[str, Path, None] = None,
        filter_instruments_dir: Union[str, Path, None] = None,
        filter_parquet_date_col: str = "date",
        filter_parquet_inst_col: str = "ts_code",
        force_refresh: bool = False,
        cache_label: bool = True,
        n_jobs: int = 1,
        parallel_backend: str = "threads",
        share_data: str = "none",
        show_progress: bool = False,
        progress_every: int = 10,
        **kwargs,
    ):
        if infer_processors is None:
            infer_processors = []
        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)

        data_loader = {
            "class": "Alpha158CustomDataLoader",
            "module_path": "qlib_strategy_core.handlers.alpha_158_custom",
            "kwargs": {
                "config": {
                    "feature": self.get_feature_config(),
                    "label": kwargs.pop("label", self.get_label_config()),
                },
                "filter_pipe": filter_pipe,
                "freq": freq,
                "inst_processors": inst_processors,
                "use_cache": use_cache,
                "cache_root": cache_root,
                "cache_filtered": cache_filtered,
                "cache_label": cache_label,
                "filter_market": filter_market,
                "filter_parquet": filter_parquet,
                "filter_instruments_dir": filter_instruments_dir,
                "filter_parquet_date_col": filter_parquet_date_col,
                "filter_parquet_inst_col": filter_parquet_inst_col,
                "force_refresh": force_refresh,
                "n_jobs": n_jobs,
                "parallel_backend": parallel_backend,
                "share_data": share_data,
                "show_progress": show_progress,
                "progress_every": progress_every,
            },
        }
        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            process_type=process_type,
            **kwargs,
        )

    def get_feature_config(self):
        conf = {
            "kbar": {},
            "price": {
                "windows": [0],
                "feature": ["OPEN", "HIGH", "LOW", "VWAP"],
            },
            "rolling": {},
        }
        return Alpha158DL.get_feature_config(conf)

    def get_label_config(self):
        return ["Ref($close, -2)/Ref($close, -1) - 1"], ["LABEL0"]
