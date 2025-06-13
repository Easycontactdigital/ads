#!/usr/bin/env python3
"""
CSV Data Loader for Google Ads Creative Generator
Loads real campaign data from CSV files instead of using Google Ads API
"""

import pandas as pd
import numpy as np
from datetime import datetime
import re
import os

def read_csv_with_encoding(file_path, **kwargs):
    """
    Read CSV file with proper encoding handling for Google Ads/Excel exports.
    
    Args:
        file_path: Path to the CSV file
        **kwargs: Additional arguments to pass to pd.read_csv
        
    Returns:
        DataFrame or raises ValueError if file cannot be read
    """
    # Common encodings used by Google Ads and Excel exports
    encodings_to_try = ['utf-16', 'utf-8-sig', 'utf-8', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings_to_try:
        # Try different separators (comma and tab)
        separators_to_try = ['\t', ',']  # Tab first since Google Ads often exports as TSV
        
        for sep in separators_to_try:
            try:
                # Try with different CSV parsing options for Google Ads exports
                df = pd.read_csv(
                    file_path, 
                    encoding=encoding,
                    sep=sep,
                    quotechar='"',
                    skipinitialspace=True,
                    on_bad_lines='skip',  # Skip problematic lines
                    **kwargs
                )
                print(f"   Successfully read with {encoding} encoding and '{sep}' separator")
                return df
            except (UnicodeDecodeError, UnicodeError, pd.errors.EmptyDataError):
                continue
            except Exception as e:
                # If it's a parsing error, try with different options
                if 'tokenizing' in str(e).lower() or 'expected' in str(e).lower():
                    try:
                        # Try with more flexible parsing
                        df = pd.read_csv(
                            file_path,
                            encoding=encoding,
                            sep=sep,
                            quotechar='"',
                            doublequote=True,
                            skipinitialspace=True,
                            on_bad_lines='skip',
                            engine='python',  # Use Python engine for more flexibility
                            **kwargs
                        )
                        print(f"   Successfully read with {encoding} encoding and '{sep}' separator (flexible parsing)")
                        return df
                    except:
                        continue
                # If it's not an encoding or parsing error, re-raise it
                elif 'codec' not in str(e).lower() and 'encoding' not in str(e).lower():
                    if sep == separators_to_try[-1]:  # Only raise on last separator attempt
                        raise e
                continue
    
    raise ValueError(f"Could not read {file_path} with any supported encoding: {encodings_to_try}")

def list_campaign_csvs(data_dir="data"):
    """
    List all CSV files in the data directory.
    
    Returns:
        List of CSV filenames
    """
    try:
        if not os.path.exists(data_dir):
            print(f"❌ Data directory '{data_dir}' does not exist")
            return []
        
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        print(f"✅ Found {len(csv_files)} campaign CSV files")
        return sorted(csv_files)
    except Exception as e:
        print(f"❌ Error listing campaign CSVs: {e}")
        return []

def load_campaign_asset_data(campaign_file, data_dir="data"):
    """
    Load and process asset data from a specific campaign CSV file.
    Focuses only on Headlines and Descriptions with performance ratings.
    
    Args:
        campaign_file: Name of the CSV file
        data_dir: Directory containing the CSV files
        
    Returns:
        DataFrame with asset performance data
    """
    try:
        file_path = os.path.join(data_dir, campaign_file)
        campaign_name = campaign_file.replace('.csv', '').replace('_', ' ')
        
        # Read the CSV with proper encoding handling for Google Ads exports
        df = read_csv_with_encoding(file_path, skiprows=2)
        
        # Debug: Print column names to see what we're working with
        print(f"   Available columns: {list(df.columns)}")
        
        # Filter for Headlines and Descriptions only
        if 'Asset type' not in df.columns:
            print(f"⚠️ 'Asset type' column not found in {campaign_file}")
            print(f"   Available columns: {list(df.columns)}")
            return pd.DataFrame()
            
        asset_df = df[df['Asset type'].isin(['Headline', 'Description'])].copy()
        
        if asset_df.empty:
            print(f"⚠️ No Headlines or Descriptions found in {campaign_file}")
            print(f"   Asset types found: {df['Asset type'].unique()}")
            return pd.DataFrame()
        
        # Assign performance category based on which column is 100%
        def get_performance_category(row):
            for category in ['Best', 'Good', 'Low', 'Learning', 'Unrated']:
                if pd.notna(row.get(category)) and str(row[category]).strip() == '100.00%':
                    return category
            return 'Unrated'
        
        asset_df['performance_category'] = asset_df.apply(get_performance_category, axis=1)
        
        # Clean and process metrics
        def clean_numeric(value):
            if pd.isna(value):
                return 0
            try:
                # Remove commas and quotes
                clean_val = str(value).replace(',', '').replace('"', '')
                return float(clean_val) if clean_val.replace('.', '').isdigit() else 0
            except:
                return 0
        
        def clean_percentage(value):
            if pd.isna(value):
                return 0.0
            try:
                clean_val = str(value).replace('%', '').strip()
                return float(clean_val) / 100 if clean_val.replace('.', '').isdigit() else 0.0
            except:
                return 0.0
        
        # Process the asset data
        processed_assets = []
        
        for _, row in asset_df.iterrows():
            asset_data = {
                'campaign_name': campaign_name,
                'asset': row['Asset'],
                'asset_type': row['Asset type'],
                'performance_category': row['performance_category'],
                'impressions': clean_numeric(row.get('Impr.', 0)),
                'clicks': clean_numeric(row.get('Clicks', 0)),
                'ctr': clean_percentage(row.get('CTR', '0%')),
                'cost': clean_numeric(row.get('Cost', 0)),
                'conversions': clean_numeric(row.get('Conversions', 0)),
                'conversion_rate': clean_percentage(row.get('Conv. rate', '0%')),
                'cost_per_conversion': clean_numeric(row.get('Cost / conv.', 0)),
            }
            processed_assets.append(asset_data)
        
        result_df = pd.DataFrame(processed_assets)
        print(f"✅ Loaded {len(result_df)} assets from {campaign_file} ({campaign_name})")
        
        # Show breakdown by category
        category_counts = result_df['performance_category'].value_counts()
        print(f"   Performance breakdown: {dict(category_counts)}")
        
        return result_df
        
    except Exception as e:
        print(f"❌ Error loading campaign asset data from {campaign_file}: {e}")
        return pd.DataFrame()

def get_campaign_summary(campaign_file, data_dir="data"):
    """
    Get a summary of a campaign's performance.
    
    Args:
        campaign_file: Name of the CSV file
        data_dir: Directory containing the CSV files
        
    Returns:
        Dictionary with campaign summary
    """
    try:
        asset_data = load_campaign_asset_data(campaign_file, data_dir)
        
        if asset_data.empty:
            return {}
        
        summary = {
            'campaign_name': asset_data['campaign_name'].iloc[0],
            'total_assets': len(asset_data),
            'headlines': len(asset_data[asset_data['asset_type'] == 'Headline']),
            'descriptions': len(asset_data[asset_data['asset_type'] == 'Description']),
            'best_assets': len(asset_data[asset_data['performance_category'] == 'Best']),
            'good_assets': len(asset_data[asset_data['performance_category'] == 'Good']),
            'learning_assets': len(asset_data[asset_data['performance_category'] == 'Learning']),
            'low_assets': len(asset_data[asset_data['performance_category'] == 'Low']),
            'total_impressions': asset_data['impressions'].sum(),
            'total_clicks': asset_data['clicks'].sum(),
            'avg_ctr': asset_data['ctr'].mean(),
            'total_conversions': asset_data['conversions'].sum(),
            'avg_conversion_rate': asset_data['conversion_rate'].mean(),
            'total_cost': asset_data['cost'].sum(),
        }
        
        return summary
        
    except Exception as e:
        print(f"❌ Error getting campaign summary for {campaign_file}: {e}")
        return {}

def load_ads_data_from_csv(file_path="example-ads-data.csv"):
    """
    Load and process ads data from CSV file.
    
    Returns:
        DataFrame with processed ad data
    """
    try:
        # Read the CSV with proper encoding handling for Google Ads exports
        df = read_csv_with_encoding(file_path, skiprows=2)
        
        # Clean up the data
        df = df[df['Ad status'] == 'Enabled']  # Only enabled ads
        df = df.dropna(subset=['Campaign'])    # Remove rows without campaign
        
        # Process the data to match our expected format
        processed_ads = []
        
        for _, row in df.iterrows():
            # Extract headlines (up to 15)
            headlines = []
            for i in range(1, 16):
                headline_col = f'Headline {i}'
                if headline_col in df.columns and pd.notna(row[headline_col]):
                    headlines.append(row[headline_col])
            
            # Extract descriptions (up to 4)  
            descriptions = []
            for i in range(1, 5):
                desc_col = f'Description {i}'
                if desc_col in df.columns and pd.notna(row[desc_col]):
                    descriptions.append(row[desc_col])
            
            # Also check the 'Description' column
            if pd.notna(row.get('Description')):
                descriptions.append(row['Description'])
            
            ad_data = {
                'ad_id': row['Ad ID'] if pd.notna(row.get('Ad ID')) else f"ad_{len(processed_ads)}",
                'ad_type': row['Ad type'],
                'status': 'ENABLED',
                'ad_group_id': row['Ad group ID'] if pd.notna(row.get('Ad group ID')) else f"ag_{len(processed_ads)}",
                'ad_group_name': row['Ad group'],
                'campaign_id': row['Campaign ID'] if pd.notna(row.get('Campaign ID')) else f"camp_{len(processed_ads)}",
                'campaign_name': row['Campaign'],
                'headlines': '|'.join(headlines) if headlines else '',
                'descriptions': '|'.join(descriptions) if descriptions else '',
                'final_url': row.get('Final URL', ''),
            }
            
            processed_ads.append(ad_data)
        
        result_df = pd.DataFrame(processed_ads)
        print(f"✅ Loaded {len(result_df)} ads from CSV")
        return result_df
        
    except Exception as e:
        print(f"❌ Error loading ads data: {e}")
        return pd.DataFrame()

def load_performance_data_from_csv(file_path="example-ads-data.csv"):
    """
    Load and process performance data from CSV file.
    
    Returns:
        DataFrame with performance metrics
    """
    try:
        # Read the CSV with proper encoding handling for Google Ads exports
        df = read_csv_with_encoding(file_path, skiprows=2)
        
        # Clean up the data
        df = df[df['Ad status'] == 'Enabled']  # Only enabled ads
        df = df.dropna(subset=['Campaign'])    # Remove rows without campaign
        
        # Process performance metrics
        processed_performance = []
        
        for _, row in df.iterrows():
            # Convert metrics to proper types
            clicks = row.get('Clicks', 0)
            impressions = row.get('Impr.', 0)
            ctr = row.get('CTR', '0%')
            cost = row.get('Cost', 0)
            conversions = row.get('Conversions', 0)
            conv_rate = row.get('Conv. rate', '0%')
            
            # Clean percentage values
            if isinstance(ctr, str):
                ctr = float(ctr.replace('%', '')) / 100 if ctr.replace('%', '').replace('.', '').isdigit() else 0
            if isinstance(conv_rate, str):
                conv_rate = float(conv_rate.replace('%', '')) / 100 if conv_rate.replace('%', '').replace('.', '').isdigit() else 0
            
            perf_data = {
                'ad_id': row['Ad ID'] if pd.notna(row.get('Ad ID')) else f"ad_{len(processed_performance)}",
                'ad_type': row['Ad type'],
                'ad_group_id': row['Ad group ID'] if pd.notna(row.get('Ad group ID')) else f"ag_{len(processed_performance)}",
                'ad_group_name': row['Ad group'],
                'campaign_id': row['Campaign ID'] if pd.notna(row.get('Campaign ID')) else f"camp_{len(processed_performance)}",
                'campaign_name': row['Campaign'],
                'impressions': int(impressions) if pd.notna(impressions) and str(impressions).isdigit() else 0,
                'clicks': int(clicks) if pd.notna(clicks) and str(clicks).isdigit() else 0,
                'ctr': float(ctr) if pd.notna(ctr) else 0,
                'cost': float(cost) if pd.notna(cost) and str(cost).replace('.', '').isdigit() else 0,
                'conversions': float(conversions) if pd.notna(conversions) else 0,
                'conversion_rate': float(conv_rate) if pd.notna(conv_rate) else 0,
                'cost_per_conversion': (float(cost) / float(conversions)) if pd.notna(conversions) and float(conversions) > 0 else 0,
            }
            
            processed_performance.append(perf_data)
        
        result_df = pd.DataFrame(processed_performance)
        print(f"✅ Loaded performance data for {len(result_df)} ads from CSV")
        return result_df
        
    except Exception as e:
        print(f"❌ Error loading performance data: {e}")
        return pd.DataFrame()

def load_asset_performance_from_csv(file_path="example-ad-asset-breakdown.csv"):
    """
    Load asset-level performance data from CSV file.
    
    Returns:
        DataFrame with asset performance data
    """
    try:
        # Read the CSV with proper encoding handling for Google Ads exports
        df = read_csv_with_encoding(file_path, skiprows=2)
        
        # Clean up the data
        df = df[df['Asset status'] == 'Enabled']  # Only enabled assets
        df = df.dropna(subset=['Asset'])         # Remove rows without asset
        
        # Filter for headlines and descriptions only
        content_assets = df[df['Asset type'].isin(['Headline', 'Description'])]
        
        processed_assets = []
        
        for _, row in df.iterrows():
            # Parse impressions properly
            impressions = 0
            if pd.notna(row.get('Impr.')):
                try:
                    impressions = int(float(row['Impr.']))
                except (ValueError, TypeError):
                    impressions = 0
            
            asset_data = {
                'asset': row['Asset'],
                'asset_type': row['Asset type'],
                'performance': row.get('Performance', 'Unknown'),
                'impressions': impressions,
                'level': row.get('Level', 'Unknown'),
                'status': row['Status'],
            }
            processed_assets.append(asset_data)
        
        result_df = pd.DataFrame(processed_assets)
        print(f"✅ Loaded {len(result_df)} assets from CSV")
        return result_df
        
    except Exception as e:
        print(f"❌ Error loading asset data: {e}")
        return pd.DataFrame()

def get_campaigns_from_csv(file_path="example-ads-data.csv"):
    """
    Extract campaign list from CSV data.
    
    Returns:
        DataFrame with campaign information
    """
    try:
        # Read the CSV with proper encoding handling for Google Ads exports
        df = read_csv_with_encoding(file_path, skiprows=2)
        
        # Get unique campaigns
        campaigns = df[['Campaign', 'Campaign ID']].drop_duplicates()
        campaigns = campaigns.dropna()
        
        processed_campaigns = []
        for _, row in campaigns.iterrows():
            campaign_data = {
                'campaign_id': row['Campaign ID'] if pd.notna(row.get('Campaign ID')) else f"camp_{len(processed_campaigns)}",
                'campaign_name': row['Campaign'],
                'status': 'ENABLED',
                'channel_type': 'SEARCH',
                'bidding_strategy': 'TARGET_CPA',
            }
            processed_campaigns.append(campaign_data)
        
        result_df = pd.DataFrame(processed_campaigns)
        print(f"✅ Found {len(result_df)} campaigns in CSV")
        return result_df
        
    except Exception as e:
        print(f"❌ Error extracting campaigns: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    # Test the data loading
    print("🧪 Testing CSV Data Loading")
    print("=" * 50)
    
    # Test campaign listing
    campaigns = list_campaign_csvs()
    print(f"Available campaigns: {campaigns}")
    
    if campaigns:
        # Test loading first campaign
        first_campaign = campaigns[0]
        print(f"\n📊 Testing with {first_campaign}")
        
        asset_data = load_campaign_asset_data(first_campaign)
        print(f"Asset data shape: {asset_data.shape}")
        
        if not asset_data.empty:
            print(f"Sample assets:")
            print(asset_data[['asset', 'asset_type', 'performance_category', 'impressions']].head())
        
        summary = get_campaign_summary(first_campaign)
        print(f"\nCampaign summary: {summary}")
    
    # Test legacy functions
    ads_df = load_ads_data_from_csv()
    print(f"Ads DataFrame shape: {ads_df.shape}")
    
    perf_df = load_performance_data_from_csv()
    print(f"Performance DataFrame shape: {perf_df.shape}")
    
    asset_df = load_asset_performance_from_csv()
    print(f"Assets DataFrame shape: {asset_df.shape}")
    
    campaigns_df = get_campaigns_from_csv()
    print(f"Campaigns DataFrame shape: {campaigns_df.shape}")
    
    if not ads_df.empty:
        print(f"\nSample ad headlines: {ads_df.iloc[0]['headlines'][:100]}...")
    
    if not perf_df.empty:
        print(f"Average CTR: {perf_df['ctr'].mean():.2%}")
        print(f"Total conversions: {perf_df['conversions'].sum():.1f}") 