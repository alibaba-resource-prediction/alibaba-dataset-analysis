import pandas as pd
import seaborn as sns
import dask.dataframe as dd
import matplotlib.pyplot as plt
from config import OUTPUT_PATH, setup_cpu_cluster

def main():
    with setup_cpu_cluster() as cluster:
        client = cluster.get_client()
        print(client.dashboard_link)

        df = pd.read_csv(OUTPUT_PATH + "ms_most_data.csv")
        df = df[df["count"] == df["count"].min()]
        random_ms = df["msname"].sample(30).tolist()

        datapath = "/home/mpds/data/bronze/table=MSMetrics"
        ms_metrics = dd.read_parquet(
            datapath,
            split_row_groups=True,
            columns=["timestamp", "day", "msname", "msinstanceid"],
            filters=[("msname", "in", random_ms)],
        )

        for ms in random_ms:
            ms_df = ms_metrics[ms_metrics["msname"] == ms].compute()
            if ms_df.empty:
                print("no data at all for", ms)
                continue


            counts = (
                ms_df.groupby(["msinstanceid", "day"])
                .size()
                .reset_index(name="count")
                .sort_values("day")
            )

            print(f"{ms}: {counts['msinstanceid'].nunique()} instances")

            sns.lineplot(data=counts, x="day", y="count", hue="msinstanceid", marker="o")
            plt.title(f"{ms}: datapoints per day per instance")
            plt.xlabel("day")
            plt.ylabel("datapoint count")
            plt.savefig(f"plots/instance_counts_{ms}.png")
            plt.clf()  # clear between plots

if __name__ == "__main__":
    main()