from langgraph import Node, Graph, Agent


# ----------------------------
# Input Node
# ----------------------------
class InputNode(Node):
    def run(self, data):
        """
        Pass the input data to the next node.
        """
        return data


# ----------------------------
# Processing Node
# ----------------------------
class ProcessingNode(Node):
    def run(self, data):
        """
        Calculate profit, CAC, and % changes.
        """
        today = data["today"]
        yesterday = data["yesterday"]

        daily_revenue = today["revenue"]
        daily_cost = today["cost"]
        num_customers = today["customers"]

        yesterday_revenue = yesterday["revenue"]
        yesterday_cost = yesterday["cost"]

        profit = daily_revenue - daily_cost
        cac = daily_cost / num_customers if num_customers else 0

        prev_cac = yesterday_cost / yesterday["customers"] if yesterday["customers"] else 0

        revenue_change_pct = ((daily_revenue - yesterday_revenue) / yesterday_revenue * 100) if yesterday_revenue else 0
        cost_change_pct = ((daily_cost - yesterday_cost) / yesterday_cost * 100) if yesterday_cost else 0

        cac_change_pct = ((cac - prev_cac) / prev_cac * 100) if prev_cac else 0

        data["metrics"] = {
            "profit": profit,
            "cac": cac,
            "cac_change_pct": cac_change_pct,
            "revenue_change_pct": revenue_change_pct,
            "cost_change_pct": cost_change_pct
        }

        return data


# ----------------------------
# Recommendation Node
# ----------------------------
class RecommendationNode(Node):
    def run(self, data):
        """
        Generate recommendations based on metrics.
        """
        profit = data["metrics"]["profit"]
        cac_change_pct = data["metrics"]["cac_change_pct"]
        revenue_change_pct = data["metrics"]["revenue_change_pct"]

        recommendations = []
        alerts = []

        if profit < 0:
            recommendations.append("Reduce costs if profit is negative.")
            alerts.append("Profit is negative.")

        if cac_change_pct > 20:
            recommendations.append("Review marketing campaigns — CAC increased significantly.")
            alerts.append("CAC increased more than 20%.")

        if revenue_change_pct > 0:
            recommendations.append("Consider increasing advertising budget if sales are growing.")

        return {
            "profit_status": "Profit" if profit >= 0 else "Loss",
            "alerts": alerts,
            "recommendations": recommendations
        }


# ----------------------------
# Build & Run Graph
# ----------------------------
def build_agent():
    input_node = InputNode()
    processing_node = ProcessingNode()
    recommendation_node = RecommendationNode()

    graph = Graph() \
        .add_node("input", input_node) \
        .add_node("process", processing_node) \
        .add_node("recommend", recommendation_node) \
        .add_edge("input", "process") \
        .add_edge("process", "recommend")

    return Agent(graph)


if __name__ == "__main__":
    agent = build_agent()

    sample_input = {
        "today": {"revenue": 1200, "cost": 800, "customers": 40},
        "yesterday": {"revenue": 1000, "cost": 600, "customers": 40}
    }

    output = agent.run("input", sample_input)
    print("Agent Output:")
    print(output)
