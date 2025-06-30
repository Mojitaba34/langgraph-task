class SystemPrompts:
    RECOMMENDATION = """
        Your task is to generate actionable recommendations based on the processed metrics.
        
        Examples of recommendations:
        - "Reduce costs if profit is negative"
        - "Review marketing campaigns if Customer Acquisition Cost (CAC) increased significantly"
        - "Consider increasing advertising budget if sales are growing"
        
        Use clear, concise advice tailored to the given data.
        """
    CALCULATE_DATA = """
                Task:   
                You are a processing node in a data pipeline. Your job is to:
                
                1. Calculate daily profit:
                    daily_profit = daily_revenue - daily_cost
                
                2. Calculate percentage changes compared to the previous day:
                    Revenue change (%) = ((today_revenue - yesterday_revenue) / yesterday_revenue) * 100
                    Cost change (%) = ((today_cost - yesterday_cost) / yesterday_cost) * 100
                
                3. Calculate Customer Acquisition Cost (CAC):
                    CAC = daily_cost / number_of_customers
                
                   Then check if today’s CAC has increased more than 20% compared to yesterday:
                    CAC change (%) = ((today_CAC - yesterday_CAC) / yesterday_CAC) * 100
                
                   If CAC change (%) > 20%, flag it.
        """


class UserPrompts:
    INPUT_DATA = """
            Here is the input data:
            {data}
            """
