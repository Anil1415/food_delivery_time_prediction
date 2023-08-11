












class handle_column_name:

    def __init__(self) -> None:
        pass

    def lower_column_names(self):
        for features in df.columns:
            df.columns = df.columns.str.lower()
        return df.columns
    def fill_column_names(self):
        for feature in df.columns:
            df.columns = df.columns.str.replace(" ","_")
        return df.columns