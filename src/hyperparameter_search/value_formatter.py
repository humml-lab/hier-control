from typing import Callable


class ValueFormatter:
    @staticmethod
    def _formatted_value(value) -> str:
        if isinstance(value, dict):
            return ValueFormatter.dict_method()(value)
        elif isinstance(value, list):
            return ValueFormatter.list_method()(value)
        elif isinstance(value, tuple):
            return ValueFormatter.tuple_method()(value)
        elif isinstance(value, bool):
            return ValueFormatter.bool_method()(value)
        else:
            return ValueFormatter.safe_str(value)

    @staticmethod
    def dict(keys: list[str] = None, separator: str = "_") -> Callable[[dict], str]:
        def format_dict(d: dict) -> str:
            selected_keys = keys if keys is not None else sorted(d.keys())
            parts = []
            for key in selected_keys:
                if key not in d:
                    continue
                value = d[key]
                if isinstance(value, dict):
                    nested_str = format_dict(value)
                    parts.append(f"{key}={nested_str}")
                else:
                    parts.append(f"{key}={ValueFormatter._formatted_value(value)}")
            return f"({separator.join(parts)})"

        return lambda x: format_dict(x)

    @staticmethod
    def list(separator: str = "-") -> Callable[[list], str]:
        def format_list(lst: list) -> str:
            if not lst:
                return "empty"
            return f"({separator.join([ValueFormatter._formatted_value(item) for item in lst])})"

        return lambda x: format_list(x)

    @staticmethod
    def tuple(separator: str = "-") -> Callable[[tuple], str]:
        list_formatter = ValueFormatter.list(separator)
        return lambda x: list_formatter(list(x))

    @staticmethod
    def bool() -> Callable[[bool], str]:
        return lambda x: "T" if x else "F"

    @staticmethod
    def safe_str(value) -> str:
        if value is None:
            return "null"
        elif isinstance(value, bool):
            return "T" if value else "F"
        elif isinstance(value, float):
            return f"{value:.2f}".replace(".", "p")
        elif isinstance(value, int):
            return f"{value:d}"
        else:
            s = str(value).replace("/", "-").replace("\\", "-").replace(":", "_")
            return s.replace(" ", "").replace("\n", "").replace("\t", "")

    @staticmethod
    def value_scheduling(keys: list = None) -> Callable[[dict], str]:
        """
        Formats value_scheduling dict with short keys for hyperparameters['initial'] values.
        Example: "astart=2e-4_aent=1e-10_acons=1_ascont=0_arec=0.001"
        """
        key_map = {
            "alpha_gtf": "agtf",
            "alpha_entropy": "aent",
            "alpha_consistency": "acons",
            "alpha_subject_continuity": "ascont",
            "alpha_reconstruction": "arec",
            "alpha_enc_l1": "aencl1",
            "alpha_dec_l1": "adecl1",
            "alpha_MAR": "amar",
            "alpha_dsr": "adsr",
        }

        # Wenn keine keys angegeben -> alle in der Map
        selected_keys = keys if keys is not None else list(key_map.keys())

        def format_value_scheduling(d: dict) -> str:
            parts = []
            for key in selected_keys:
                if key in d and "hyperparameters" in d[key]:
                    init_val = d[key]["hyperparameters"].get("initial", None)
                    if init_val is not None and float(init_val) != 0.0:
                        short = key_map.get(key, key)  # fallback: original key
                        parts.append(f"{short}={ValueFormatter.safe_str(init_val)}")
            return "(" + "_".join(parts) + ")"

        return lambda x: format_value_scheduling(x)
