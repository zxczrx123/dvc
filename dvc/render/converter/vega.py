import os
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from funcy import first, last

from dvc.exceptions import DvcException
from dvc.render import FILENAME_FIELD, INDEX_FIELD, VERSION_FIELD

from . import Converter


class FieldNotFoundError(DvcException):
    def __init__(self, expected_field, found_fields):
        found_str = ", ".join(found_fields)
        super().__init__(
            f"Could not find provided field ('{expected_field}') "
            f"in data fields ('{found_str}')."
        )


def _lists(blob: Union[Dict, List]) -> Iterable[List]:
    if isinstance(blob, list):
        yield blob
    else:
        for _, value in blob.items():
            if isinstance(value, dict):
                yield from _lists(value)
            elif isinstance(value, list):
                yield value


def _file_field(*args):
    for axis_def in args:
        if axis_def is not None:
            for file, val in axis_def.items():
                if isinstance(val, str):
                    yield file, val
                elif isinstance(val, list):
                    for field in val:
                        yield file, field


def _find(
    filename: str,
    field: str,
    data_series: List[Tuple[str, str, Any]],
):
    for data_file, data_field, data in data_series:
        if data_file == filename and data_field == field:
            return data_file, data_field, data
    return None


def _verify_field(file2datapoints: Dict[str, List], filename: str, field: str):
    if filename in file2datapoints:
        datapoint = first(file2datapoints[filename])
        if field not in datapoint:
            raise FieldNotFoundError(field, datapoint.keys())


def _get_xs(properties: Dict, file2datapoints: Dict[str, List[Dict]]):
    x = properties.get("x", None)
    if x is not None and isinstance(x, dict):
        for filename, field in _file_field(x):
            _verify_field(file2datapoints, filename, field)
            yield filename, field


def _get_ys(properties, file2datapoints: Dict[str, List[Dict]]):
    y = properties.get("y", None)
    if y is not None:
        for filename, field in _file_field(y):
            _verify_field(file2datapoints, filename, field)
            yield filename, field


def _is_datapoints(lst: List[Dict]):
    """
    check if dict keys match, datapoints with different keys mgiht lead
    to unexpected behavior
    """

    return all(isinstance(item, dict) for item in lst) and set(first(lst).keys()) == {
        key for keys in lst for key in keys
    }


def get_datapoints(file_content: Dict):
    result: List[Dict[str, Any]] = []
    for lst in _lists(file_content):
        if _is_datapoints(lst):
            for index, datapoint in enumerate(lst):
                if len(result) <= index:
                    result.append({})
                result[index].update(datapoint)
    return result


class VegaConverter(Converter):
    """
    Class that takes care of converting unspecified data blob
    (Dict or List[Dict]) into datapoints (List[Dict]).
    If some properties that are required by Template class are missing
    ('x', 'y') it will attempt to fill in the blanks.
    """

    def __init__(
        self,
        plot_id: str,
        data: Optional[Dict] = None,
        properties: Optional[Dict] = None,
    ):
        super().__init__(plot_id, data, properties)
        self.plot_id = plot_id

    def _infer_y_from_data(self):
        if self.plot_id in self.data:
            for lst in _lists(self.data[self.plot_id]):
                if all(isinstance(item, dict) for item in lst):
                    datapoint = first(lst)
                    field = last(datapoint.keys())
                    return {self.plot_id: field}

    def _infer_x_y(self):
        x = self.properties.get("x", None)
        y = self.properties.get("y", None)

        inferred_properties: Dict = {}

        # Infer x.
        if isinstance(x, str):
            inferred_properties["x"] = {}
            # If multiple y files, duplicate x for each file.
            if isinstance(y, dict):
                for file, fields in y.items():
                    # Duplicate x for each y.
                    if isinstance(fields, list):
                        inferred_properties["x"][file] = [x] * len(fields)
                    else:
                        inferred_properties["x"][file] = x
            # Otherwise use plot ID as file.
            else:
                inferred_properties["x"][self.plot_id] = x

        # If x is a dict, y must be a dict.
        elif isinstance(x, dict) and not isinstance(y, dict):
            raise DvcException(
                f"Error with {self.plot_id}: cannot specify a data source for x"
                " without a data source for y."
            )

        # Infer y.
        if y is None:
            inferred_properties["y"] = self._infer_y_from_data()
        # If y files not provided, use plot ID as file.
        elif not isinstance(y, dict):
            inferred_properties["y"] = {self.plot_id: y}

        return inferred_properties

    def _find_datapoints(self):
        result = {}
        for file, content in self.data.items():
            result[file] = get_datapoints(content)

        return result

    @staticmethod
    def infer_y_label(properties):
        y_label = properties.get("y_label", None)
        if y_label is not None:
            return y_label
        y = properties.get("y", None)
        if isinstance(y, str):
            return y
        if isinstance(y, list):
            return "y"
        if not isinstance(y, dict):
            return

        fields = {field for _, field in _file_field(y)}
        if len(fields) == 1:
            return first(fields)
        return "y"

    @staticmethod
    def infer_x_label(properties):
        x_label = properties.get("x_label", None)
        if x_label is not None:
            return x_label

        x = properties.get("x", None)
        if not isinstance(x, dict):
            return INDEX_FIELD

        fields = {field for _, field in _file_field(x)}
        if len(fields) == 1:
            return first(fields)
        return "x"

    @staticmethod
    def _props_update(xs, ys):
        props_update: Dict = {}

        xs_dict = dict(xs)
        all_x_files = set(xs_dict.keys())
        all_x_fields = set(xs_dict.values())
        # assign x field
        if all_x_fields:
            props_update["x_files"] = all_x_files
            props_update["x"] = first(all_x_fields)
        # override to unified x field name if there are different x fields
        if len(all_x_fields) > 1:
            props_update["x"] = "dvc_inferred_x_value"

        all_y_files = {file for file, field in ys}
        all_y_fields = {field for file, field in ys}
        # assign y files and field
        props_update["y_files"] = all_y_files
        props_update["y"] = first(all_y_fields)
        # override to unified y field name if there are different y fields
        if len(all_y_fields) > 1:
            props_update["y"] = "dvc_inferred_y_value"

        num_x_files = len(all_x_files)
        num_y_files = len(all_y_files)
        if num_x_files > 1 and num_x_files != num_y_files:
            raise DvcException(
                "Cannot have different number of x and y data sources. Found "
                f"{num_x_files} x and {num_y_files} y data sources."
            )

        return props_update

    @staticmethod
    def _common_prefix_len(paths):
        common_prefix_len = 0
        if len(paths) > 1:
            common_prefix_len = len(os.path.commonpath(paths))
        return common_prefix_len

    def flat_datapoints(self, revision):
        file2datapoints, properties = self.convert()

        xs = list(_get_xs(properties, file2datapoints))
        ys = list(_get_ys(properties, file2datapoints))

        props_update = self._props_update(xs, ys)

        # get common prefix to drop from file names
        all_y_files = props_update.pop("y_files", [])
        common_prefix_len = self._common_prefix_len(all_y_files)

        # fixed x values for single x
        all_x_files = props_update.pop("x_files", [])
        x_file = first(all_x_files)
        x_field = props_update.get("x", None)

        all_datapoints = []

        for y_file, y_field in ys:
            datapoints = [{**d} for d in file2datapoints.get(y_file, [])]

            # update if multiple y values
            if props_update.get("y", None) == "dvc_inferred_y_value":
                _update_from_field(
                    datapoints,
                    field="dvc_inferred_y_value",
                    source_field=y_field,
                )

            # map x to y if multiple x files
            if len(all_x_files) > 1:
                try:
                    x_file = y_file
                    x_field = dict(xs)[x_file]
                except KeyError:
                    raise DvcException(  # noqa: B904
                        f"No x value found for y data source {y_file}."
                    )
            # update x field
            if x_field:
                x_datapoints = file2datapoints.get(x_file, [])
                try:
                    _update_from_field(
                        datapoints,
                        field=props_update["x"],
                        source_datapoints=x_datapoints,
                        source_field=x_field,
                    )
                except IndexError:
                    raise DvcException(  # noqa: B904
                        f"Cannot join '{x_field}' from '{x_file}' and "
                        f"'{y_field}' from '{y_file}'. "
                        "They have to have same length."
                    )
            # assign "step" if no x provided
            else:
                props_update["x"] = INDEX_FIELD
                _update_from_index(datapoints, INDEX_FIELD)

            y_file_short = y_file[common_prefix_len:].strip("/\\")
            _update_all(
                datapoints,
                update_dict={
                    VERSION_FIELD: {
                        "revision": revision,
                        FILENAME_FIELD: y_file_short,
                        "field": y_field,
                    }
                },
            )

            all_datapoints.extend(datapoints)

        if not all_datapoints:
            return [], {}

        properties = {**properties, **props_update}

        return all_datapoints, properties

    def convert(
        self,
    ):
        """
        Convert the data. Fill necessary fields ('x', 'y') and return both
        generated datapoints and updated properties. If `x` is not provided,
        leave it as None, fronteds should handle it.

        NOTE: Studio uses this method.
              The only thing studio FE handles is filling `x` and `y`.
              `x/y_label` should be filled here.

              Datapoints are not stripped according to config, because users
              might be utilizing other fields in their custom plots.
        """
        datapoints = self._find_datapoints()
        inferred_properties = self._infer_x_y()
        properties = {**self.properties, **inferred_properties}

        properties["y_label"] = self.infer_y_label(properties)
        properties["x_label"] = self.infer_x_label(properties)

        return datapoints, properties


def _update_from_field(
    target_datapoints: List[Dict],
    field: str,
    source_datapoints: Optional[List[Dict]] = None,
    source_field: Optional[str] = None,
):
    if source_datapoints is None:
        source_datapoints = target_datapoints
    if source_field is None:
        source_field = field

    if len(source_datapoints) != len(target_datapoints):
        raise IndexError("Source and target datapoints must have the same length")

    for index, datapoint in enumerate(target_datapoints):
        source_datapoint = source_datapoints[index]
        if source_field in source_datapoint:
            datapoint[field] = source_datapoint[source_field]


def _update_from_index(datapoints: List[Dict], new_field: str):
    for index, datapoint in enumerate(datapoints):
        datapoint[new_field] = index


def _update_all(datapoints: List[Dict], update_dict: Dict):
    for datapoint in datapoints:
        datapoint.update(update_dict)
