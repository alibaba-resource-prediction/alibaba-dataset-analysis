import pandas as pd
import dask.dataframe as dd
import seaborn as sns
import matplotlib.pyplot as plt
from config import setup_cpu_cluster, OUTPUT_PATH

def main():
    with setup_cpu_cluster() as cluster:
        client = cluster.get_client()
        print(client.dashboard_link)

        datapath = "/home/mpds/data/bronze/table=MSMetrics"
        edges_path = OUTPUT_PATH + "graph_edges_full.csv"

        amount = 40          # top N destinations
        caller_amount = 10   # top callers per destination

        edges = pd.read_csv(edges_path)
        edges = edges[~edges["dm"].isin(["UNKNOWN", "UNAVAILABLE", "USER"])]
        edges = edges[~edges["um"].isin(["UNKNOWN", "UNAVAILABLE", "USER"])]

        top_ms = edges.nlargest(amount, "count")[["dm", "count"]]
        print(top_ms)

        callers_by_ms = {}
        interesting_ms = set(top_ms["dm"].tolist())
        for ms in top_ms["dm"]:
            top_callers = (
                edges[edges["dm"] == ms]
                .nlargest(caller_amount, "count")["um"]
                .tolist()
            )
            callers_by_ms[ms] = top_callers
            interesting_ms.update(top_callers)
            print(f"Top callers for {ms}:")
            for caller, count in zip(
                top_callers,
                edges[edges["dm"] == ms].nlargest(caller_amount, "count")["count"]
            ):
                print(f"  {caller}: {count}")

        interesting_ms = list(interesting_ms)
        if not interesting_ms:
            print("No MS to plot after filtering.")
            return

        all_metrics = dd.read_parquet(
            datapath,
            split_row_groups=True,
            columns=["day", "timestamp", "msname", "cpu_utilization"],
            filters=[("msname", "in", interesting_ms)],
        )

        # mean cpu_utilization per msname/timestamp
        g = (
            all_metrics.groupby(["msname", "timestamp"])["cpu_utilization"]
            .mean()
            .reset_index()
        ).compute()

        available_ms = set(g["msname"].unique())

        for ms in top_ms["dm"]:
            if ms not in available_ms:
                print(f"Skip {ms}: no metrics found for destination itself")
                continue

            entities_to_plot = [ms] + callers_by_ms.get(ms, [])
            plt.figure(figsize=(10, 6))
            found_any = False
            for entity in entities_to_plot:
                entity_data = g[g["msname"] == entity]
                if entity_data.empty:
                    continue
                found_any = True
                label = f"{entity} (target)" if entity == ms else entity
                sns.lineplot(
                    data=entity_data.sort_values("timestamp"),
                    x="timestamp",
                    y="cpu_utilization",
                    label=label,
                )
            if not found_any:
                print(f"No metrics found for {ms} or its callers")
                plt.clf()
                continue

            plt.title(f"CPU Utilization for {ms} and top {caller_amount} callers")
            plt.xlabel("timestamp")
            plt.ylabel("mean cpu_utilization")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"plots/cpu_utilization_{ms}_and_callers.png")
            plt.clf()

if __name__ == "__main__":
    main()