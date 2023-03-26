from os import name, pread

import pandas as pd
from etna.datasets.tsdataset import TSDataset
from etna.models import SeasonalMovingAverageModel
from etna.pipeline import Pipeline
from etna.transforms import (
    DensityOutliersTransform,
    LogTransform,
    TimeSeriesImputerTransform,
)


class Model:
    TARGET_COLUMNS = {
        "sold_volume": "sum_price",
        "sold_count": "cnt",
    }

    def _create_pipe(self):
        model = SeasonalMovingAverageModel()

        transforms = [
            DensityOutliersTransform(
                in_column="target", window_size=30, n_neighbors=7, distance_coef=1.9
            ),
            TimeSeriesImputerTransform(
                in_column="target", strategy="running_mean", window=3
            ),
            LogTransform(in_column="target", inplace=True),
        ]

        pipe = Pipeline(
            model=model,
            transforms=transforms,
            horizon=30,
            # step=1
        )
        return pipe

    def _predict(self, pipe):
        forecast = pipe.forecast()
        outp_df = (
            forecast.to_pandas()
            .stack()
            .reset_index()
            .drop("feature", axis=1)
            .melt("timestamp")
        )
        return outp_df

    def fit(self, data, target, path_to_save):
        ts = self._procces_input(data, self.TARGET_COLUMNS[target])
        pipe = self._create_pipe()
        pipe = pipe.fit(ts)
        pipe.save(path_to_save)


    def agg_predict(self, data: dict, pipeline_path: str, target: str) -> dict:
        """
        data: dict - data for predict
        pipeline_path: str - path to model pipeline
        target: str - sold_volume or sold_count
        """
        ts = self._procces_input(data, self.TARGET_COLUMNS[target])
        history = ts.loc["2022-9-24":]
        pipe = Pipeline.load(pipeline_path, ts=ts)

        outp_df = self._predict(pipe)
        outp_df = outp_df.rename(
            columns={
                "timestamp": "dt",
                "value": "prediction_value",
                "segment": "region_code",
            }
        )

        history = (
            history.copy()
            .stack()
            .reset_index()
            .drop("feature", axis=1)
            .melt("timestamp")
        )
        history = history.rename(
            columns={
                "timestamp": "dt",
                "value": "history_value",
                "segment": "region_code",
            }
        )
        outp_df["dt"] = outp_df["dt"].astype(str)
        history["dt"] = history["dt"].astype(str)

        return history.to_dict(orient="records") + outp_df.to_dict(orient="records")

    def manufacturer_predict(
        self, data: dict, sale_points: dict, pipeline_path: str, target: str
    ) -> dict:
        """
        data: dict - data for predict
        sale_points: dict - data about sale points
        pipeline_path: str - path to model pipeline
        target: str - sold_volume or sold_count
        """
        data = self._region_agg(data, sale_points, self.TARGET_COLUMNS[target])
        ts = self._procces_input(data, self.TARGET_COLUMNS[target], dropna=True)
        history = ts.loc["2022-9-24":]
        pipe = Pipeline.load(pipeline_path, ts=ts)
        pipe = pipe.fit(ts)
        outp_df = self._predict(pipe)
        outp_df = outp_df.rename(
            columns={
                "timestamp": "dt",
                "value": "prediction_value",
                "segment": "region_code",
            }
        )

        history = (
            history.copy()
            .stack()
            .reset_index()
            .drop("feature", axis=1)
            .melt("timestamp")
        )
        history = history.rename(
            columns={
                "timestamp": "dt",
                "value": "history_value",
                "segment": "region_code",
            }
        )
        outp_df["dt"] = outp_df["dt"].astype(str)
        history["dt"] = history["dt"].astype(str)

        return history.to_dict(orient="records") + outp_df.to_dict(orient="records")

    def volume_agg_predict(self, data: dict, pipeline_path: str) -> dict:
        """
        data: dict - data for predict
        pipeline_path: str - path to model pipeline
        """
        return self.agg_predict(
            data=data, pipeline_path=pipeline_path, target="sold_volume"
        )

    def count_agg_predict(self, data: dict, pipeline_path: str) -> dict:
        """
        data: dict - data for predict
        pipeline_path: str - path to model pipeline
        """
        return self.agg_predict(
            data=data, pipeline_path=pipeline_path, target="sold_count"
        )

    def volume_manufacturer_predict(
        self, data: dict, sale_points: dict, pipeline_path: str
    ) -> dict:
        """
        data: dict - data for predict
        sale_points: dict - data about sale points
        pipeline_path: str - path to model pipeline
        """
        return self.manufacturer_predict(
            data=data,
            sale_points=sale_points,
            pipeline_path=pipeline_path,
            target="sold_count",
        )

    def count_manufacturer_predict(
        self, data: dict, sale_points: dict, pipeline_path: str
    ) -> dict:
        """
        data: dict - data for predict
        sale_points: dict - data about sale points
        pipeline_path: str - path to model pipeline
        """
        return self.manufacturer_predict(
            data=data,
            sale_points=sale_points,
            pipeline_path=pipeline_path,
            target="sold_count",
        )

    def _procces_input(self, data, target_column, dropna=False):
        df = pd.DataFrame(data)
        df = df.rename(
            columns={
                "dt": "timestamp",
                target_column: "target",
                "region_code": "segment",
            }
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df[["timestamp", "target", "segment"]]

        df = df.dropna()
        df = df.drop_duplicates().reset_index(drop=True)

        df = (
            df.groupby(["segment", "timestamp"])
            .aggregate("sum")
            .reset_index(level=[0, 1])
        )
        # tmp = df.groupby('segment')['timestamp'].count().sort_values()
        # df = df[df['segment'].apply(lambda x: tmp[x] > 180)]

        ts = TSDataset.to_dataset(df)
        if dropna:
            ts = ts.dropna(axis=1)
        else:
            ts = ts.fillna(0)

        ts = TSDataset(ts, freq="D")

        return ts

    def _region_agg(self, data: dict, sale_points: dict, target_column: str):
        data = pd.DataFrame(data)
        sale_points = pd.DataFrame(sale_points)

        sale_points["region_code"] = sale_points["region_code"].astype(object)
        data["sum_price"] = data["price"] * data["cnt"]

        data = data.merge(sale_points, "left", "id_sp_")

        data = data.dropna()
        data = data.groupby(["region_code", "dt"])[target_column].sum().reset_index()
        data["dt"] = pd.to_datetime(data["dt"])
        data = data.sort_values("dt")
        return data
