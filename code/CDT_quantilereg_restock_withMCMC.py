import pandas as pd
import numpy as np
from statsmodels.regression.quantile_regression import QuantReg
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Load data
data_df = pd.read_csv('/project/workspace/data/trans_data.csv')
current_stock_df = pd.read_csv('/project/workspace/data/current_stock.csv')
product_importance_df = pd.read_csv('/project/workspace/data/product_importance.csv')

# 1. Cost-Benefit Analysis
def calculate_utility(expected_return, variance, service_level_achievement, holding_cost, stockout_cost, ordering_cost, lambda_1, lambda_2, lambda_3):
    total_cost = holding_cost + stockout_cost + ordering_cost
    return expected_return - (lambda_1 * variance) + (lambda_2 * service_level_achievement) - (lambda_3 * total_cost)

# Function to analyze counterfactual scenarios
def counterfactual_analysis(current_inventory, reorder_point, safety_stock, demand_forecast, service_level_target):
    # Calculate current service level based on demand forecast
    demand_variance = np.std(demand_forecast)  # Sample variance
    expected_demand = np.mean(demand_forecast)
    
    # Calculate service level as the probability of meeting demand
    service_level = (current_inventory + safety_stock) / expected_demand

    # Calculate impact of increasing reorder point
    new_reorder_point = reorder_point + 10  # Example increase
    new_service_level = (current_inventory + safety_stock) / (expected_demand + 10)  # Assumed demand increase

    # Calculate impact of reducing safety stock
    reduced_safety_stock = max(0, safety_stock - 5)  # Example reduction
    reduced_service_level = (current_inventory + reduced_safety_stock) / expected_demand

    return {
        'current_service_level': service_level,
        'new_service_level_with_increased_reorder': new_service_level,
        'new_service_level_with_reduced_safety_stock': reduced_service_level
    }

# Dynamic adjustment function
def update_utility_for_product(product_utility, service_level_data, desired_service_level):
    if service_level_data < desired_service_level:
        # Increase emphasis on meeting service levels
        product_utility *= 1.2  # Example adjustment factor
    return product_utility

# 2. Decision Tree Analysis
def decision_tree_analysis(data, features, target):
    # Train decision tree model
    model = DecisionTreeClassifier(max_depth=3)
    model.fit(data[features], data[target])

    # Plot decision tree
    # plt.figure(figsize=(12,8))
    # plot_tree(model, feature_names=features, class_names=[target], filled=True)
    # plt.title("Decision Tree for Inventory Decisions")
    # plt.show()

# 3. Scenario Planning with Monte Carlo Simulation
def monte_carlo_simulation(num_simulations, demand_mean, demand_std, reorder_point, safety_stock):
    results = []
    for _ in range(num_simulations):
        # Simulate demand
        simulated_demand = np.random.normal(demand_mean, demand_std)
        # Check if stockout occurs
        if simulated_demand > (reorder_point + safety_stock):
            results.append(1)  # Stockout occurs
        else:
            results.append(0)  # No stockout
    return np.mean(results)  # Proportion of stockouts

def process_product(data, product_name, lead_time=7, service_level_factor=1.645, quantiles=[0.25, 0.5, 0.75], plot=False):
    # Fetch product importance and current inventory level from the CSV files
    try:
        product_importance = product_importance_df.loc[product_importance_df['Description'] == product_name, 'importance'].values[0]
    except IndexError:
        product_importance = 1.0  # Default importance if not found in CSV

    try:
        current_inventory_level = current_stock_df.loc[current_stock_df['Description'] == product_name, 'current_stock'].values[0]
    except IndexError:
        current_inventory_level = 50  # Default inventory if not found in CSV

    # Filter data for the specific product
    product_data = data[data['Description'] == product_name].copy()
    # Ensure that 'trans_date' is in datetime format
    product_data['trans_date'] = pd.to_datetime(product_data['trans_date'])
    product_data = product_data.set_index('trans_date').resample('D').sum().reset_index()  # Resample to daily
    product_data['day_of_week'] = product_data['trans_date'].dt.dayofweek
    product_data['month'] = product_data['trans_date'].dt.month
    product_data['lag_1'] = product_data['Quantity'].shift(1)  # AR lag 1 day
    product_data['lag_7'] = product_data['Quantity'].shift(7)  # AR lag 7 days
    product_data['ma_7'] = product_data['Quantity'].rolling(window=7).mean().shift(1)  # MA effect: 7-day moving average
    product_data.dropna(inplace=True)  # Drop NaN values due to lagging

    # Train-test split based on real data
    train = product_data[product_data['trans_date'] < '2011-12-01']
    test = product_data[(product_data['trans_date'] >= '2011-12-01') & (product_data['trans_date'] < '2011-12-09')]

    # Perform Quantile Regression for each quantile level
    predictions = pd.DataFrame({'trans_date': test['trans_date']})
    for q in quantiles:
        model = QuantReg(train['Quantity'], train[['day_of_week', 'month', 'lag_1', 'lag_7', 'ma_7']])
        res = model.fit(q=q)
        predictions[f'quantile_{int(q * 100)}'] = res.predict(test[['day_of_week', 'month', 'lag_1', 'lag_7', 'ma_7']])

    # Summarize quantile predictions for lead time demand and safety stock
    quantile_25 = predictions['quantile_25'].mean()
    quantile_50 = predictions['quantile_50'].mean()
    quantile_75 = predictions['quantile_75'].mean()

    # Lead Time Demand (D_L): Sum median demand over the lead time
    daily_median_demand = quantile_50
    lead_time_demand = daily_median_demand * lead_time

    # Safety Stock (SS): Adjusted using quantile-based variance
    demand_variance = quantile_75 - quantile_25
    safety_stock = service_level_factor * demand_variance

    # Calculate costs for utility calculation
    holding_cost = 100  # Example holding cost
    stockout_cost = 200  # Example stockout cost
    ordering_cost = 50  # Example ordering cost
    
    # Utility Calculation (incorporating product importance)
    lambda_1, lambda_2, lambda_3 = 0.5, 1.0, 1.0
    expected_return = quantile_50
    utility = calculate_utility(
        expected_return=expected_return,
        variance=demand_variance,
        service_level_achievement=0.90,  # Example achievement for utility calculation
        holding_cost=holding_cost,
        stockout_cost=stockout_cost,
        ordering_cost=ordering_cost,
        lambda_1=lambda_1,
        lambda_2=lambda_2,
        lambda_3=lambda_3
    )

    # Restocking Recommendation
    reorder_point = lead_time_demand + safety_stock
    restocking_amount = max(0, reorder_point - current_inventory_level)

    # Counterfactual Analysis
    demand_forecast = np.random.normal(loc=daily_median_demand, scale=demand_variance, size=100)  # Simulated demand forecast
    counterfactual_results = counterfactual_analysis(current_inventory_level, reorder_point, safety_stock, demand_forecast, service_level_target=0.90)

    # Dynamic Adjustment for Utility
    desired_service_level = 0.95
    service_level_data = counterfactual_results['current_service_level']  # Use current service level from analysis
    updated_utility = update_utility_for_product(utility, service_level_data, desired_service_level)

    # Plot results if plot=True
    if plot:
        plt.figure(figsize=(14, 8))

        # Line plot of historical demand with quantile predictions
        plt.plot(product_data['trans_date'], product_data['Quantity'], label='Historical Demand', color='blue')
        plt.plot(predictions['trans_date'], predictions['quantile_25'], label='Quantile 25%', linestyle='--', color='orange')
        plt.plot(predictions['trans_date'], predictions['quantile_50'], label='Quantile 50% (Median)', linestyle='-', color='green')
        plt.plot(predictions['trans_date'], predictions['quantile_75'], label='Quantile 75%', linestyle='--', color='red')
        plt.axhline(y=reorder_point, color='purple', linestyle='-', label='Reorder Point')
        plt.axhline(y=current_inventory_level, color='black', linestyle=':', label='Current Inventory Level')
        plt.xlabel('Date')
        plt.ylabel('Quantity')
        plt.title(f'Demand and Quantile Forecasting for {product_name}')
        plt.legend()
        plt.show()

        # Bar plot of quantile values for the product
        plt.figure(figsize=(8, 6))
        plt.bar(['Quantile 25%', 'Quantile 50% (Median)', 'Quantile 75%'], [quantile_25, quantile_50, quantile_75], color=['orange', 'green', 'red'])
        plt.title(f'Quantile Demand Levels for {product_name}')
        plt.ylabel('Quantity')
        plt.show()

        # Bar plot comparing current inventory vs reorder point
        plt.figure(figsize=(8, 6))
        plt.bar(['Current Inventory', 'Reorder Point'], [current_inventory_level, reorder_point], color=['grey', 'purple'])
        plt.title(f'Inventory vs Reorder Point for {product_name}')
        plt.ylabel('Quantity')
        plt.show()

    # Run Decision Tree Analysis
    decision_data = pd.DataFrame({
        'Demand Spike': [1, 1, 0, 0, 1, 0, 1, 0],
        'Supplier Delay': [0, 1, 1, 0, 0, 1, 1, 0],
        'Reorder Inventory': [1, 0, 0, 1, 1, 0, 1, 0]
    })

    features = ['Demand Spike', 'Supplier Delay']
    target = 'Reorder Inventory'
    decision_tree_analysis(decision_data, features, target)

    # Monte Carlo simulation for stockout probability
    num_simulations = 10000
    stockout_probability = monte_carlo_simulation(num_simulations, daily_median_demand, demand_variance, reorder_point, safety_stock)
    print(f"Estimated Stockout Probability for {product_name}: {stockout_probability:.2%}")

    # Return the result as a dictionary
    return {
        'product_name': product_name,
        'utility': updated_utility,
        'restocking_amount': restocking_amount,
        'current_inventory': current_inventory_level,
        'quantile_25_total': quantile_25,
        'quantile_50_total': quantile_50,
        'quantile_75_total': quantile_75,
        'lead_time_demand': lead_time_demand,
        'safety_stock': safety_stock,
        'reorder_point': reorder_point,
        'counterfactual_results': counterfactual_results
    }

if __name__ == "__main__":
    # Initialize a list to store results
    results = []

    # Run the function for each product in your data and store results
    for product in data_df['Description'].unique():
        product_result = process_product(data_df, product, lead_time=7, service_level_factor=1.645, plot=False)
        results.append(product_result)

    # Display Results in descending order of utility
    print("Product Utilities (Ranked):")
    for result in sorted(results, key=lambda x: x['utility'], reverse=True):
        print(f"{result['product_name']}: Utility = {result['utility']:.2f}, Restocking Amount = {result['restocking_amount']:.2f}, Current Inventory = {result['current_inventory']}")
