#!/usr/bin/env python3
"""
Generate comprehensive sales dataset with 10,000 rows
Maintains logical business relationships and realistic patterns
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_comprehensive_sales_data():
    """Generate 10,000 rows of realistic sales data with proper business logic"""
    
    print("🏭 Generating comprehensive sales dataset...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Configuration
    n_rows = 10000
    start_date = datetime(2015, 1, 1)
    
    # Base data structures
    salespeople = ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank', 'Grace', 'Henry', 'Ivy', 'Jack']
    regions = ['North', 'South', 'East', 'West', 'Central']
    product_categories = ['Software', 'Hardware', 'Services', 'Consulting', 'Support']
    
    # Salesperson skill levels (affects performance)
    salesperson_skills = {
        'Alice': 0.95, 'Bob': 0.85, 'Charlie': 0.90, 'Diana': 0.80, 'Eve': 0.88,
        'Frank': 0.75, 'Grace': 0.92, 'Henry': 0.78, 'Ivy': 0.87, 'Jack': 0.83
    }
    
    # Regional market factors
    regional_factors = {
        'North': 1.15, 'South': 0.95, 'East': 1.10, 'West': 1.05, 'Central': 0.90
    }
    
    # Product category base prices and margins
    product_info = {
        'Software': {'base_price': 450, 'margin': 0.25, 'seasonality': 1.0},
        'Hardware': {'base_price': 800, 'margin': 0.15, 'seasonality': 1.2},
        'Services': {'base_price': 350, 'margin': 0.35, 'seasonality': 0.9},
        'Consulting': {'base_price': 1200, 'margin': 0.40, 'seasonality': 0.8},
        'Support': {'base_price': 200, 'margin': 0.50, 'seasonality': 1.1}
    }
    
    data = []
    
    for i in range(n_rows):
        if i % 1000 == 0:
            print(f"   Generated {i} rows...")
        
        # Generate date (distributed across ~10 years)
        days_offset = int(np.random.exponential(365 * 2.5))  # Weighted toward recent dates
        date = start_date + timedelta(days=days_offset)
        
        # Ensure we don't go beyond 2024
        if date.year > 2024:
            date = datetime(2024, 12, 31) - timedelta(days=np.random.randint(0, 365))
        
        # Basic attributes
        salesperson = random.choice(salespeople)
        region = random.choice(regions)
        product_category = random.choice(product_categories)
        
        # Get factors
        skill_factor = salesperson_skills[salesperson]
        regional_factor = regional_factors[region]
        product_info_current = product_info[product_category]
        
        # Seasonal factors
        month = date.month
        seasonal_factor = 1.0
        if month in [11, 12]:  # Holiday boost
            seasonal_factor = 1.25
        elif month in [1, 2]:  # Post-holiday dip
            seasonal_factor = 0.85
        elif month in [6, 7, 8]:  # Summer stability
            seasonal_factor = 1.05
        
        seasonal_factor *= product_info_current['seasonality']
        
        # Economic index (varies over time with some trends)
        base_economic = 100
        year_trend = (date.year - 2015) * 0.8  # Gradual growth
        economic_cycle = 5 * np.sin(2 * np.pi * (date.year - 2015) / 7)  # 7-year cycle
        economic_noise = np.random.normal(0, 2)
        economic_index = base_economic + year_trend + economic_cycle + economic_noise
        
        # Market competition (inversely related to economic conditions)
        market_competition = max(1.0, 6.0 - (economic_index - 95) / 10 + np.random.normal(0, 0.5))
        
        # Marketing spend (base + seasonal + random + skill-based)
        base_marketing = 45000
        marketing_seasonal = base_marketing * (seasonal_factor - 1) * 0.3
        marketing_skill = base_marketing * (skill_factor - 0.8) * 0.5
        marketing_random = np.random.normal(0, 8000)
        marketing_spend = max(10000, base_marketing + marketing_seasonal + marketing_skill + marketing_random)
        
        # Digital marketing (portion of total marketing)
        digital_ratio = 0.25 + 0.15 * (date.year - 2015) / 9  # Increasing over time
        digital_marketing = marketing_spend * digital_ratio * (0.8 + 0.4 * np.random.random())
        
        # Lead generation (strongly correlated with marketing spend)
        lead_base = marketing_spend * 0.004  # Base conversion rate
        lead_skill_bonus = lead_base * (skill_factor - 0.8) * 1.5
        lead_regional_bonus = lead_base * (regional_factor - 1) * 0.8
        lead_noise = np.random.normal(0, lead_base * 0.15)  # Reduced noise for stronger correlation
        lead_generation = max(50, lead_base + lead_skill_bonus + lead_regional_bonus + lead_noise)
        
        # Website traffic (correlated with digital marketing)
        traffic_base = digital_marketing * 0.5
        traffic_noise = np.random.normal(0, traffic_base * 0.3)
        website_traffic = max(1000, traffic_base + traffic_noise)
        
        # Social media engagement (related to digital marketing)
        social_base = digital_marketing * 0.15
        social_noise = np.random.normal(0, social_base * 0.4)
        social_media_engagement = max(100, social_base + social_noise)
        
        # Training hours (varies by salesperson and time)
        base_training = 40
        skill_training = (1 - skill_factor) * 30  # Lower skill = more training
        time_training = (date.year - 2015) * 2  # Increasing training over time
        training_noise = np.random.normal(0, 8)
        training_hours = max(10, base_training + skill_training + time_training + training_noise)
        
        # Product quality score (improves over time, varies by category)
        quality_base = 8.0
        quality_time_trend = (date.year - 2015) * 0.05  # Gradual improvement
        quality_category_factor = {'Software': 0.2, 'Hardware': -0.1, 'Services': 0.1, 'Consulting': 0.3, 'Support': 0.0}[product_category]
        quality_noise = np.random.normal(0, 0.3)
        product_quality_score = np.clip(quality_base + quality_time_trend + quality_category_factor + quality_noise, 5, 10)
        
        # Customer satisfaction (influenced by training, product quality, skill)
        satisfaction_base = 7.5
        satisfaction_training = (training_hours - 40) * 0.025  # Stronger training effect
        satisfaction_skill = (skill_factor - 0.8) * 6  # Stronger skill effect
        satisfaction_quality = (product_quality_score - 8.0) * 0.3  # Add quality effect
        satisfaction_noise = np.random.normal(0, 0.4)  # Reduced noise
        customer_satisfaction = np.clip(satisfaction_base + satisfaction_training + satisfaction_skill + satisfaction_quality + satisfaction_noise, 1, 10)
        
        # Brand awareness (builds over time, influenced by marketing)
        brand_base = 6.5
        brand_time = (date.year - 2015) * 0.15
        brand_marketing = (marketing_spend - 45000) / 45000 * 0.5
        brand_noise = np.random.normal(0, 0.4)
        brand_awareness = np.clip(brand_base + brand_time + brand_marketing + brand_noise, 3, 10)
        
        # Conversion rate (influenced by satisfaction, quality, competition)
        conversion_base = 0.15
        conversion_satisfaction = (customer_satisfaction - 7.5) * 0.008
        conversion_quality = (product_quality_score - 8.0) * 0.004
        conversion_competition = (4.0 - market_competition) * 0.008
        conversion_noise = np.random.normal(0, 0.015)  # Reduced noise for more consistent relationship
        conversion_rate = np.clip(conversion_base + conversion_satisfaction + conversion_quality + conversion_competition + conversion_noise, 0.08, 0.25)
        
        # Sales volume (influenced by leads, conversion rate, seasonal factors)
        volume_base = lead_generation * conversion_rate * seasonal_factor * 5  # Scale up for realistic volumes
        volume_regional = volume_base * (regional_factor - 1) * 0.3
        volume_noise = np.random.normal(0, max(10, volume_base * 0.1))  # Reduced noise
        sales_volume = max(100, volume_base + volume_regional + volume_noise)
        
        # Ensure no infinite or NaN values
        if not np.isfinite(sales_volume):
            sales_volume = 500  # Default fallback
        
        # Price (base price with market and competition adjustments)
        price_base = product_info_current['base_price']
        price_competition = price_base * (market_competition - 3.5) * 0.02  # Higher competition = lower price
        price_quality = price_base * (product_quality_score - 8.0) * 0.03  # Higher quality = higher price
        price_noise = np.random.normal(0, price_base * 0.05)
        price = max(price_base * 0.7, price_base + price_competition + price_quality + price_noise)
        
        # Competitor price (related but different)
        competitor_price = price * (0.95 + 0.1 * np.random.random()) + np.random.normal(0, 20)
        
        # Revenue (sales volume * price)
        revenue = sales_volume * price
        
        # Ensure no infinite or NaN values
        if not np.isfinite(revenue):
            revenue = 250000  # Default fallback
        
        # Customer acquisition cost
        acquired_customers = max(1, lead_generation * conversion_rate)
        customer_acquisition_cost = marketing_spend / acquired_customers
        
        # Ensure reasonable bounds
        customer_acquisition_cost = np.clip(customer_acquisition_cost, 50, 2000)
        
        # Profit margin
        base_margin = product_info_current['margin']
        margin_efficiency = (skill_factor - 0.8) * 0.1  # Better salespeople = better margins
        margin_competition = (4.0 - market_competition) * 0.02  # Less competition = better margins
        margin_noise = np.random.normal(0, 0.03)
        profit_margin = np.clip(base_margin + margin_efficiency + margin_competition + margin_noise, 0.05, 0.6)
        
        # Customer retention (influenced by satisfaction and quality)
        retention_base = 0.85
        retention_satisfaction = (customer_satisfaction - 7.5) * 0.02
        retention_quality = (product_quality_score - 8.0) * 0.01
        retention_noise = np.random.normal(0, 0.05)
        customer_retention = np.clip(retention_base + retention_satisfaction + retention_quality + retention_noise, 0.6, 0.98)
        
        # Market share (builds over time, influenced by performance)
        share_base = 0.18
        share_time = (date.year - 2015) * 0.003  # Gradual growth
        share_performance = (revenue - 400000) / 400000 * 0.02  # Performance-based
        share_noise = np.random.normal(0, 0.01)
        market_share = np.clip(share_base + share_time + share_performance + share_noise, 0.1, 0.35)
        
        # Create row
        row = {
            'Date': date.strftime('%Y-%m-%d'),
            'Salesperson': salesperson,
            'Region': region,
            'Product_Category': product_category,
            'Marketing_Spend': round(marketing_spend, 2),
            'Lead_Generation': round(lead_generation, 0),
            'Sales_Volume': round(sales_volume, 0),
            'Revenue': round(revenue, 2),
            'Customer_Satisfaction': round(customer_satisfaction, 2),
            'Training_Hours': round(training_hours, 1),
            'Market_Competition': round(market_competition, 2),
            'Price': round(price, 2),
            'Customer_Acquisition_Cost': round(customer_acquisition_cost, 2),
            'Profit_Margin': round(profit_margin, 3),
            'Seasonal_Factor': round(seasonal_factor, 2),
            'Economic_Index': round(economic_index, 2),
            'Digital_Marketing': round(digital_marketing, 2),
            'Social_Media_Engagement': round(social_media_engagement, 0),
            'Website_Traffic': round(website_traffic, 0),
            'Conversion_Rate': round(conversion_rate, 3),
            'Customer_Retention': round(customer_retention, 3),
            'Product_Quality_Score': round(product_quality_score, 2),
            'Brand_Awareness': round(brand_awareness, 2),
            'Competitor_Price': round(competitor_price, 2),
            'Market_Share': round(market_share, 3)
        }
        
        data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Sort by date for better time series analysis
    df = df.sort_values('Date').reset_index(drop=True)
    
    print(f"✅ Generated {len(df)} rows of sales data")
    return df

def analyze_data_relationships(df):
    """Analyze the generated data to verify relationships"""
    
    print("\n📊 Analyzing data relationships...")
    
    # Key correlations to check
    correlations = {
        'Marketing_Spend vs Lead_Generation': df['Marketing_Spend'].corr(df['Lead_Generation']),
        'Lead_Generation vs Sales_Volume': df['Lead_Generation'].corr(df['Sales_Volume']),
        'Sales_Volume vs Revenue': df['Sales_Volume'].corr(df['Revenue']),
        'Training_Hours vs Customer_Satisfaction': df['Training_Hours'].corr(df['Customer_Satisfaction']),
        'Digital_Marketing vs Website_Traffic': df['Digital_Marketing'].corr(df['Website_Traffic']),
        'Customer_Satisfaction vs Customer_Retention': df['Customer_Satisfaction'].corr(df['Customer_Retention']),
        'Product_Quality_Score vs Brand_Awareness': df['Product_Quality_Score'].corr(df['Brand_Awareness']),
        'Market_Competition vs Profit_Margin': df['Market_Competition'].corr(df['Profit_Margin']),
    }
    
    print("\n🔗 Key Correlations:")
    for relationship, correlation in correlations.items():
        strength = "Strong" if abs(correlation) >= 0.7 else "Moderate" if abs(correlation) >= 0.4 else "Weak"
        print(f"   {relationship}: {correlation:.3f} ({strength})")
    
    # Data summary
    print(f"\n📋 Data Summary:")
    print(f"   Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"   Salespeople: {df['Salesperson'].nunique()} ({', '.join(df['Salesperson'].unique())})")
    print(f"   Regions: {df['Region'].nunique()} ({', '.join(df['Region'].unique())})")
    print(f"   Product Categories: {df['Product_Category'].nunique()} ({', '.join(df['Product_Category'].unique())})")
    print(f"   Revenue range: ${df['Revenue'].min():,.0f} - ${df['Revenue'].max():,.0f}")
    print(f"   Average revenue: ${df['Revenue'].mean():,.0f}")

if __name__ == "__main__":
    print("🏭 Comprehensive Sales Data Generator")
    print("=" * 50)
    
    # Generate data
    df = generate_comprehensive_sales_data()
    
    # Analyze relationships
    analyze_data_relationships(df)
    
    # Save to CSV
    output_file = "sales_data.csv"
    df.to_csv(output_file, index=False)
    print(f"\n💾 Data saved to {output_file}")
    
    print(f"\n🎉 Dataset ready for dashboard demonstration!")
    print(f"   • {len(df)} rows of realistic business data")
    print(f"   • Strong causal relationships for analysis")
    print(f"   • Time series patterns for forecasting")
    print(f"   • Multiple categories for visualization")
    print(f"   • Realistic business logic throughout")