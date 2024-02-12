# %%
import pandas as _pd
import os as _os
import sys as _sys
import fnmatch as _fnmatch
import copy as _copy
import re as _re

_pd.set_option("display.max_columns", None)
_pd.set_option("future.no_silent_downcasting", True)
from typing import Callable as _Callable


class Data:
    def __init__(self, data: _pd.DataFrame):
        self.__data = data

    def __repr__(self):
        return self.__data.__repr__()

    def __str__(self):
        return self.__data.__str__()

    def __len__(self):
        return len(self.__data.data)

    def __iter__(self):
        return self.__data.__iter__()

    @property
    def data(self):
        return self.__data

    @property
    def shape(self):
        return self.__data.shape

    @property
    def columns(self):
        return self.__data.columns

    def __generate_headers(self, height: int = 1):
        df = self.__data
        if df.shape[0] > height:
            columns = None
            headers = (
                _pd.DataFrame(df.iloc[:height, :], dtype=object)
                .ffill(axis=1)
                .ffill(axis=0)
            )

            if height > 1:
                columns = _pd.MultiIndex.from_frame(
                    headers.T.astype(str),
                    names=list(range(height)),
                )
            elif height == 1:
                columns = _pd.Index(headers.T.iloc[:, 0])
            else:
                raise Exception("Invalid height")

        df = df.iloc[height:, :]
        df.columns = columns
        return df

    def to_pivoted_dataframe(
        self,
        column_width: int = 1,
        header_row_height: int | None = None,
        var_names: list[str] = ["TIME_PERIOD"],
        value_name: str = "OBS_VALUE",
        id_names: list[str] = ["Component"],
        remove_null_values: bool = False,
        remove_null_labels: bool = True,
    ):
        if column_width != len(id_names):
            raise Exception("id_names must be the same length as width")

        if header_row_height and header_row_height > 0:
            df = self.__generate_headers(height=header_row_height)
        else:
            df = self.__data

        if len(df.columns):
            cols = list(df.columns)
            if type(df.columns) == _pd.MultiIndex:
                for index in range(len(cols[0]) - 1):
                    cols[index] = (id_names[index],) * len(cols[index])
                df.columns = _pd.MultiIndex.from_tuples(cols)
            else:
                cols[:column_width] = id_names
                df.columns = cols

        placeholder = 909090909

        if not remove_null_values:
            df.iloc[:, column_width:] = (
                df.iloc[:, column_width:]
                .fillna(value=placeholder)
                .reset_index(drop=True)
            )

        df = (
            df.melt(
                id_vars=list(df.columns[:column_width]),
                value_vars=list(df.columns[column_width:]),
                value_name=value_name,
            )
            .dropna(subset=[value_name])
            .reset_index(drop=True)
        )

        df.loc[:, value_name] = (
            df.loc[:, value_name]
            .replace(placeholder, _pd.NA)
            .apply(_pd.to_numeric, errors="coerce", downcast="float")
        )

        length = 0

        if var_names:
            if isinstance(var_names, (list, tuple)):
                length = len(var_names)
            elif isinstance(var_names, str):
                length = 1

        cols = list(df.columns)
        cols[:column_width] = id_names

        if length:
            cols[column_width : length + column_width] = var_names
        df.columns = cols

        if remove_null_labels:
            df = df[~df.iloc[:, :column_width].isnull().sum(axis=1).astype(bool)]

        return df

    def to_dataframe(self, header_row_height=None):
        if header_row_height and header_row_height > 0:
            return self.__generate_headers(height=header_row_height)
        return self.__data

    def remove_empty_rows(self):
        return Data(self.__data.dropna(how="all").reset_index(drop=True))

    def remove_empty_columns(self):
        df = self.__data.dropna(axis=1, how="all")
        cols = df.columns
        has_no_col_names = all(isinstance(x, int) for x in cols)

        if has_no_col_names:
            df.columns = list(range(len(df.columns)))

        return Data(df)

    def remove_empty_rows_and_columns(self):
        return self.remove_empty_rows().remove_empty_columns()

    def set_bounds(
        self,
        top_left_callback: _Callable[[any], bool] | tuple[int, int],
        top_right_callback: _Callable[[any], bool] | tuple[int, int] | None = None,
        bottom_left_callback: _Callable[[any], bool] | tuple[int, int] | None = None,
        bottom_right_callback: _Callable[[any], bool] | tuple[int, int] | None = None,
    ):
        return Bounds(self.__data).set_bounds(
            top_left_callback,
            top_right_callback,
            bottom_left_callback,
            bottom_right_callback,
        )


class Bounds:
    def __init__(self, data: _pd.DataFrame = None):
        self.__data = data if len(data.index) else _pd.DataFrame()
        self.__bounds = []
        self.__max_rows = None
        self.__max_columns = None
        self.__top_row_offset = 0
        self.__right_column_offset = 0
        self.__bottom_row_offset = 0
        self.__left_column_offset = 0

    def __check_cell_callback(
        self, value: _Callable[[any], bool] | tuple[int, int] | None
    ):
        df = self.__data
        if callable(value):
            return df[df.map(lambda x: value(x))].stack().index.tolist()
        if type(value) == tuple:
            return [(value[0], value[1])]
        return None

    def __safe_get(self, main_coordinated, array):
        if not array:
            return None

        for c in array:
            if c[1] >= main_coordinated[1] and c[0] >= main_coordinated[0]:
                return c

    def set_bounds(
        self,
        top_left_callback: _Callable[[any], bool] | tuple[int, int],
        top_right_callback: _Callable[[any], bool] | tuple[int, int] | None = None,
        bottom_left_callback: _Callable[[any], bool] | tuple[int, int] | None = None,
        bottom_right_callback: _Callable[[any], bool] | tuple[int, int] | None = None,
    ):
        top_left_positions = self.__check_cell_callback(top_left_callback)
        top_right_positions = self.__check_cell_callback(top_right_callback)
        bottom_left_positions = self.__check_cell_callback(bottom_left_callback)
        bottom_right_positions = self.__check_cell_callback(bottom_right_callback)

        for index, position in enumerate(top_left_positions):
            top_left = position
            bottom_left = self.__safe_get(top_left, bottom_left_positions)
            top_right = self.__safe_get(top_left, top_right_positions)
            bottom_right = self.__safe_get(top_left, bottom_right_positions)

            if bottom_left is None:
                for index_c, tl in enumerate(top_left_positions, start=index):
                    if tl[1] == position[1] and tl[0] > position[0]:
                        bottom_left = (
                            min(tl[0] - 1, self.__data.shape[0]),
                            position[1],
                        )
                        break
                else:
                    bottom_left = (self.__data.shape[0], position[1])

            if top_right is None:
                for index_c, tl in enumerate(top_left_positions, start=index):
                    if tl[0] == position[0] and tl[1] > position[1]:
                        top_right = (position[0], min(tl[1] - 1, self.__data.shape[1]))
                        break
                else:
                    top_right = (position[0], self.__data.shape[1])

            if bottom_right is None:
                bottom_right = (bottom_left[0], top_right[1])

            self.__bounds.append((top_left, top_right, bottom_left, bottom_right))

        return self

    def offset(
        self,
        top_row: int = 0,
        right_column: int = 0,
        bottom_row: int = 0,
        left_column: int = 0,
    ):
        self.__top_row_offset = top_row
        self.__right_column_offset = right_column
        self.__bottom_row_offset = bottom_row
        self.__left_column_offset = left_column
        return self

    def set_max(self, max_rows: int = None, max_columns: int = None):
        if max_rows:
            self.__max_rows = max_rows
        if max_columns:
            self.__max_columns = max_columns
        return self

    def generate_datasets(self):
        datasets: list[Data] = []

        for bound in self.__bounds:
            top_left_row = max(bound[0][0] + self.__top_row_offset, 0)
            top_left_column = max(bound[0][1] + self.__left_column_offset, 0)

            brr = bound[3][0]
            brc = bound[3][1]

            if self.__max_rows:
                brr = top_left_row + self.__max_rows
            if self.__max_columns:
                brc = top_left_column + self.__max_columns

            bottom_right_row = min(
                brr + self.__bottom_row_offset + 1, self.__data.shape[0]
            )
            bottom_right_column = min(
                brc + self.__right_column_offset, self.__data.shape[1]
            )

            top_left = (top_left_row, top_left_column)
            bottom_right = (bottom_right_row, bottom_right_column)

            df_ = self.__data.iloc[
                top_left[0] : bottom_right[0], top_left[1] : bottom_right[1]
            ]

            datasets.append(Data(df_.reset_index(drop=True)))

        return datasets


class Worksheet:
    def __init__(self, title: str, data: _pd.DataFrame):
        self.title = title
        self.data = Data(data)


class Workbook:
    def __init__(self, file_name: str, ws: dict[str, _pd.DataFrame], alias: str = ""):
        self.file_name = file_name
        self.alias = alias
        self.worksheets: list[Worksheet] = []
        self.ws_count = 0

        for sheet, value in ws.items():
            self.worksheets.append(Worksheet(title=sheet, data=value))

        self.ws_count = len(self.worksheets)


class Parser:
    def __init__(self):
        self.workbooks: list[Workbook] = []
        self.wb_count = 0
        self.index = 0
        self.__type = None
        self.__valid_file_types = [".xlsx", ".xlsm", ".xltx", ".xltm", ".csv", ".txt"]
        self.__replacements = None
        self.__field_master: _pd.DataFrame = None
        self.__df_collections: dict[str, _pd.DataFrame] = {}
        self.__data_collections: dict[str, list] = {}

    def __get_file_data(
        self, file_path: str, workbooks_to_ignore: list[str] = None
    ) -> Workbook:
        if not self.__is_valid_file(file_path, workbooks_to_ignore):
            return self

        self.index = self.index + 1
        if self.__type == "excel":
            dfs = _pd.read_excel(file_path, sheet_name=None, header=None)
        elif self.__type == "csv":
            dfs = {_os.path.basename(file_path): _pd.read_csv(file_path, header=None)}
        else:
            raise Exception("Invalid file type")

        self.workbooks.append(Workbook(file_path, dfs, f"T{self.index}"))

        return self

    def __is_valid_file(
        self,
        file_path: str,
        ignore: list[str] = None,
    ) -> bool:
        is_valid_file_type = any(
            file_path.endswith(ext) for ext in self.__valid_file_types
        )

        file_type = file_path.split(".").pop()

        if "xl" in file_type:
            self.__type = "excel"
        elif "csv" in file_type or "txt" in file_type:
            self.__type = "csv"

        if not is_valid_file_type:
            return False
        if not _fnmatch.fnmatch(file_path, "*~$*"):
            if ignore:
                for item in ignore:
                    if _fnmatch.fnmatch(file_path, f"*{item}*"):
                        return False

            return True
        return False

    def __reload_parser(self, workbooks: list[Workbook]):
        self.workbooks = _copy.deepcopy(workbooks)
        self.wb_count = len(self.workbooks)
        self.index = self.wb_count
        return self

    def __relative_path_to_absolute(self, path: str):
        if not _os.path.isabs(path):
            cwd = _sys.path[0]
            return _os.path.join(cwd, path)
        return path

    @staticmethod
    def extract_values(df: _pd.DataFrame):
        def func(value):
            if type(value) == str:
                if _re.search(r"\[.*\]", value):
                    return value.split("[")[1].split("]")[0]

            return value

        return df.map(func)

    def get_final_df(
        self, collection_name: str = None, attach_field_master: bool = True
    ):
        if not collection_name in self.__df_collections:
            raise Exception("Collection not found")

        df = self.__df_collections[collection_name]

        if self.__field_master is None:
            print("Field master not found, returning dataset only")
            return df

        if not attach_field_master:
            return df

        self.__replacements = list(
            self.__field_master.columns.get_level_values(1).unique()
        )
        self.__replacements.append("ws_title")
        final_dataset = _pd.concat([self.__field_master, df], axis=1)
        final_dataset = final_dataset.reindex(columns=self.__replacements)
        final_dataset = self.extract_values(final_dataset)
        return final_dataset

    def get_final_data(self, collection_name: str):
        return self.__data_collections[collection_name]

    def load_field_master(self, file_path: str):
        field_master = _pd.read_csv(_os.path.abspath(file_path))
        self.__field_master = field_master.pivot(columns=["Field"])
        return self

    def concat_df(self, collection_name: str, df: _pd.DataFrame):
        if collection_name in self.__df_collections:
            self.__df_collections[collection_name] = _pd.concat(
                [self.__df_collections[collection_name], df], axis=0, ignore_index=True
            )
        else:
            self.__df_collections[collection_name] = df

    def concat_data(self, collection_name: str, data: any):
        self.__data_collections[collection_name] = data

    def load_files(
        self,
        folder_path: str,
        include_subfolders: bool,
        workbooks_to_ignore: list[str] = [],
    ):
        folder_path = self.__relative_path_to_absolute(folder_path)

        if not _os.path.exists(folder_path):
            raise Exception("Folder not found")

        if include_subfolders:
            for path, subdirs, files in _os.walk(folder_path):
                for name in files:
                    file_path = _os.path.join(path, name)
                    self.__get_file_data(file_path, workbooks_to_ignore)
        else:
            for file in _os.listdir(folder_path):
                file_path = _os.path.join(folder_path, file)
                self.__get_file_data(file_path, workbooks_to_ignore)

        self.wb_count = len(self.workbooks)

        return self

    def load_file(self, file_path: str):
        file_path = self.__relative_path_to_absolute(file_path)
        if not _os.path.exists(file_path):
            raise Exception("File not found")
        else:
            self.__get_file_data(file_path)

        self.wb_count = len(self.workbooks)

        return self

    def pipe_workbooks(
        self, callback: _Callable[[Workbook, int, list[Workbook]], Workbook | None]
    ):
        new_parser = Parser().__reload_parser(self.workbooks)
        for index, wb in enumerate(new_parser.workbooks.copy()):
            new_wb = callback(wb, index, new_parser.workbooks)
            if new_wb is not None:
                new_parser.workbooks[index] = new_wb

        return new_parser
