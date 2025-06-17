"""
Google Ads Creative Generator - Streamlit App
Main interface for generating new ad creatives using Google Ads API data analysis and OpenAI
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
import os
import sys
import json
import logging
from datetime import datetime, timedelta

# Add src directory to path
sys.path.append('src')

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # This will output to terminal
        logging.FileHandler('ads_generator.log')  # Also log to file
    ]
)
logger = logging.getLogger(__name__)

# Import our custom modules
try:
    from creative_generator import CreativeGenerator
    from google_ads_helper import create_client, load_config
    from google.ads.googleads.errors import GoogleAdsException
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Google Ads Creative Generator",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .account-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin: 0.5rem 0;
        cursor: pointer;
    }
    .account-card:hover {
        border-color: #1f77b4;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .performance-table {
        font-size: 0.9rem;
    }
    .performance-table th {
        background-color: #f8f9fa;
        font-weight: 600;
    }
    .performance-best {
        background-color: #d4edda;
        color: #155724;
    }
    .performance-good {
        background-color: #d1ecf1;
        color: #0c5460;
    }
    .performance-learning {
        background-color: #fff3cd;
        color: #856404;
    }
    .performance-low {
        background-color: #f8d7da;
        color: #721c24;
    }
    .filter-button {
        margin: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'account_selection'
    if 'google_ads_client' not in st.session_state:
        st.session_state.google_ads_client = None
    if 'accounts_structure' not in st.session_state:
        st.session_state.accounts_structure = None
    if 'selected_account' not in st.session_state:
        st.session_state.selected_account = None
    if 'campaigns_data' not in st.session_state:
        st.session_state.campaigns_data = None
    if 'selected_campaign' not in st.session_state:
        st.session_state.selected_campaign = None
    if 'assets_data' not in st.session_state:
        st.session_state.assets_data = None
    if 'asset_filter' not in st.session_state:
        st.session_state.asset_filter = 'All'
    if 'creative_insights' not in st.session_state:
        st.session_state.creative_insights = None
    if 'generated_creative' not in st.session_state:
        st.session_state.generated_creative = None
    if 'openai_api_key' not in st.session_state:
        st.session_state.openai_api_key = os.getenv('OPENAI_API_KEY', '')
    if 'insights_generated' not in st.session_state:
        st.session_state.insights_generated = False
    if 'creative_insights' not in st.session_state:
        st.session_state.creative_insights = None
    if 'processed_assets' not in st.session_state:
        st.session_state.processed_assets = None
    if 'suggested_target_audience' not in st.session_state:
        st.session_state.suggested_target_audience = ''
    if 'suggested_brief' not in st.session_state:
        st.session_state.suggested_brief = ''
    if 'replacement_recommendations' not in st.session_state:
        st.session_state.replacement_recommendations = None

def load_google_ads_client():
    """Load Google Ads client and account structure."""
    if st.session_state.google_ads_client is None:
        try:
            # Load configuration
            config = load_config()
            if not config:
                st.error("❌ Google Ads configuration not found. Please check google_ads_config.json")
                return False
            
            # Create client
            client, _ = create_client(config)
            if not client:
                st.error("❌ Failed to create Google Ads client")
                return False
            
            st.session_state.google_ads_client = client
            
            # Load account structure
            try:
                with open('complete_account_structure.json', 'r') as f:
                    st.session_state.accounts_structure = json.load(f)
            except FileNotFoundError:
                st.error("❌ Account structure not found. Please run account discovery first.")
                return False
            
            return True
            
        except Exception as e:
            st.error(f"❌ Error initializing Google Ads client: {e}")
            return False
    
    return True

def display_account_selection():
    """Display account selection interface."""
    st.header("🏢 Account Selection")
    st.caption("Choose a Google Ads account to analyze")
    
    if not load_google_ads_client():
        return False
    
    # Extract client accounts from structure
    client_accounts = []
    structure = st.session_state.accounts_structure.get('complete_structure', {})
    
    for manager_id, manager_data in structure.items():
        # Direct client accounts under manager
        for client in manager_data.get('client_accounts', []):
            if not client.get('manager', False) and client.get('status') == 'ENABLED':
                client_accounts.append({
                    'id': client['id'],
                    'name': client['name'],
                    'manager': manager_data['manager_info']['name']
                })
        
        # Client accounts under sub-managers
        for sub_manager_id, sub_manager_data in manager_data.get('sub_managers', {}).items():
            for client in sub_manager_data.get('client_accounts', []):
                if not client.get('manager', False) and client.get('status') == 'ENABLED':
                    client_accounts.append({
                        'id': client['id'],
                        'name': client['name'],
                        'manager': f"{manager_data['manager_info']['name']} > {sub_manager_data['manager_info']['name']}"
                    })
    
    if not client_accounts:
        st.error("❌ No enabled client accounts found")
        return False
    
    # Remove duplicates based on account ID
    seen_ids = set()
    unique_client_accounts = []
    for account in client_accounts:
        if account['id'] not in seen_ids:
            unique_client_accounts.append(account)
            seen_ids.add(account['id'])
    
    # Group accounts by category for better organization
    forbes_accounts = [acc for acc in unique_client_accounts if 'Forbes' in acc['name']]
    dollargeek_accounts = [acc for acc in unique_client_accounts if 'DollarGeek' in acc['name']]
    expertise_accounts = [acc for acc in unique_client_accounts if 'Expertise' in acc['name']]
    
    # Display accounts in organized tabs
    tab1, tab2, tab3 = st.tabs(["📊 Forbes Health", "💰 DollarGeek Health", "🎯 Expertise Health"])
    
    with tab1:
        display_account_group(forbes_accounts, "Forbes Health Accounts")
    
    with tab2:
        display_account_group(dollargeek_accounts, "DollarGeek Health Accounts")
    
    with tab3:
        display_account_group(expertise_accounts, "Expertise Health Accounts")
    
    return st.session_state.selected_account is not None

def display_account_group(accounts, group_name):
    """Display a group of accounts."""
    if not accounts:
        st.info(f"No {group_name.lower()} found")
        return
    
    st.subheader(f"{group_name} ({len(accounts)} accounts)")
    
    for i, account in enumerate(accounts):
        with st.container():
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"""
                <div class="account-card">
                    <h4>{account['name']}</h4>
                    <p><strong>Manager:</strong> {account['manager']}</p>
                    <p><strong>Account ID:</strong> {account['id']}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Use group name and index to ensure unique keys
                unique_key = f"select_{group_name.replace(' ', '_')}_{account['id']}_{i}"
                if st.button(f"Select", key=unique_key, type="primary"):
                    st.session_state.selected_account = account
                    st.session_state.campaigns_data = None
                    st.session_state.selected_campaign = None
                    st.session_state.assets_data = None
                    st.session_state.current_page = 'campaign_selection'
                    st.rerun()

def get_campaigns_for_account(account_id):
    """Get campaigns for the selected account."""
    try:
        ga_service = st.session_state.google_ads_client.get_service("GoogleAdsService")
        
        query = """
            SELECT
                campaign.id,
                campaign.name,
                campaign.status,
                campaign.advertising_channel_type,
                metrics.impressions,
                metrics.clicks,
                metrics.ctr,
                metrics.cost_micros
            FROM campaign
            WHERE campaign.status != 'REMOVED'
            ORDER BY metrics.impressions DESC
        """
        
        response = ga_service.search(customer_id=account_id, query=query)
        
        campaigns = []
        for row in response:
            campaign = row.campaign
            metrics = row.metrics
            
            campaigns.append({
                'id': campaign.id,
                'name': campaign.name,
                'status': campaign.status.name,
                'type': campaign.advertising_channel_type.name,
                'impressions': metrics.impressions,
                'clicks': metrics.clicks,
                'ctr': metrics.ctr,
                'cost': metrics.cost_micros / 1_000_000 if metrics.cost_micros else 0
            })
        
        return campaigns
        
    except GoogleAdsException as ex:
        st.error(f"❌ Error getting campaigns: {ex}")
        return []

def display_campaign_selection():
    """Display campaign selection interface with performance data."""
    if not st.session_state.selected_account:
        return False
    
    st.header("📈 Campaign Selection")
    st.caption(f"Campaigns for: {st.session_state.selected_account['name']}")
    
    # Date range selector
    st.subheader("📅 Performance Date Range")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Last 7 Days", key="date_7d"):
            st.session_state.date_range_days = 7
    with col2:
        if st.button("📊 Last 30 Days", key="date_30d"):
            st.session_state.date_range_days = 30
    with col3:
        if st.button("📊 Last 90 Days", key="date_90d"):
            st.session_state.date_range_days = 90
    
    # Initialize default date range
    if 'date_range_days' not in st.session_state:
        st.session_state.date_range_days = 30
    
    st.info(f"📅 Showing performance data for the last {st.session_state.date_range_days} days")
    
    # Load campaigns if not already loaded
    if not st.session_state.campaigns_data:
        with st.spinner("🔍 Loading campaigns..."):
            campaigns = get_campaigns_for_account(st.session_state.selected_account['id'])
            st.session_state.campaigns_data = campaigns
    
    campaigns_df = pd.DataFrame(st.session_state.campaigns_data)
    
    if campaigns_df.empty:
        st.warning("⚠️ No campaigns found for this account")
        return False
    
    # Display campaigns table
    st.subheader("🎯 Select Campaign")
    
    # Format the dataframe for display
    display_df = campaigns_df.copy()
    display_df['CTR %'] = (display_df['ctr'] * 100).round(2)
    display_df['Cost $'] = display_df['cost'].round(2)
    
    # Select columns to display  
    columns_to_show = ['name', 'status', 'impressions', 'clicks', 'CTR %', 'Cost $']
    
    # Style active campaigns differently
    def style_status(val):
        if val == 'ENABLED':
            return 'background-color: #d4edda; color: #155724'
        elif val == 'PAUSED':
            return 'background-color: #fff3cd; color: #856404'
        return 'background-color: #f8d7da; color: #721c24'
    
    styled_df = display_df[columns_to_show].style.map(
        style_status, subset=['status']
    ).format({
        'impressions': '{:,}',
        'clicks': '{:,}',
        'CTR %': '{:.2f}%',
        'Cost $': '${:.2f}'
    })
    
    st.dataframe(styled_df, use_container_width=True)
    
    # Campaign selection
    campaign_names = campaigns_df['name'].tolist()
    selected_campaign_name = st.selectbox(
        "Choose a campaign to analyze:",
        campaign_names,
        key="campaign_selector"
    )
    
    if selected_campaign_name and st.button("📊 Load Campaign Assets", type="primary"):
        selected_campaign = campaigns_df[campaigns_df['name'] == selected_campaign_name].iloc[0]
        st.session_state.selected_campaign = selected_campaign.to_dict()
        
        # Load assets with selected date range
        with st.spinner("🔍 Loading campaign assets..."):
            assets = get_campaign_assets(
                st.session_state.selected_campaign['id'], 
                days_back=st.session_state.date_range_days
            )
            st.session_state.assets_data = assets
        
        if assets:
            st.success(f"✅ Loaded {len(assets)} assets for {selected_campaign_name}")
            st.session_state.current_page = 'performance_analysis'
            st.rerun()
        else:
            st.warning("⚠️ No assets found for this campaign")
    
    return True

def get_campaign_assets(campaign_id, days_back=30):
    """Get asset performance with attribution from ad-level metrics based on impression share."""
    try:
        ga_service = st.session_state.google_ads_client.get_service("GoogleAdsService")
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        date_range = f"segments.date BETWEEN '{start_date.strftime('%Y-%m-%d')}' AND '{end_date.strftime('%Y-%m-%d')}'"
        
        # Step 1: Get ad-level performance data (conversions, clicks, impressions)
        ad_query = f"""
            SELECT
                campaign.name,
                ad_group.id,
                ad_group.name,
                ad_group_ad.ad.id,
                ad_group_ad.ad.name,
                metrics.impressions,
                metrics.clicks,
                metrics.ctr,
                metrics.conversions,
                metrics.cost_micros
            FROM ad_group_ad
            WHERE campaign.id = {campaign_id}
              AND campaign.status != 'REMOVED'
              AND ad_group.status != 'REMOVED'
              AND ad_group_ad.status != 'REMOVED'
              AND ad_group_ad.ad.type = 'RESPONSIVE_SEARCH_AD'
              AND {date_range}
        """
        
        # Get ad performance data
        ad_response = ga_service.search(customer_id=st.session_state.selected_account['id'], query=ad_query)
        
        # Aggregate ad performance
        ad_performance = {}
        for row in ad_response:
            campaign = row.campaign
            ad_group = row.ad_group
            ad = row.ad_group_ad.ad
            metrics = row.metrics
            
            ad_key = f"{ad_group.id}_{ad.id}"
            
            if ad_key not in ad_performance:
                ad_performance[ad_key] = {
                    'campaign_name': campaign.name,
                    'ad_group_id': str(ad_group.id),
                    'ad_group_name': ad_group.name,
                    'ad_id': str(ad.id),
                    'ad_name': ad.name or f"Ad {ad.id}",
                    'impressions': 0,
                    'clicks': 0,
                    'conversions': 0,
                    'cost_micros': 0
                }
            
            ad_performance[ad_key]['impressions'] += metrics.impressions or 0
            ad_performance[ad_key]['clicks'] += metrics.clicks or 0
            ad_performance[ad_key]['conversions'] += metrics.conversions or 0
            ad_performance[ad_key]['cost_micros'] += metrics.cost_micros or 0
        
        # Calculate CTR and CPC for ads
        for ad_data in ad_performance.values():
            if ad_data['impressions'] > 0:
                ad_data['ctr'] = ad_data['clicks'] / ad_data['impressions']
                ad_data['cost'] = ad_data['cost_micros'] / 1_000_000  # Convert micros to currency
                if ad_data['clicks'] > 0:
                    ad_data['cpc'] = ad_data['cost'] / ad_data['clicks']
                else:
                    ad_data['cpc'] = 0
            else:
                ad_data['ctr'] = 0
                ad_data['cost'] = 0
                ad_data['cpc'] = 0
        
        # Step 2: Get asset-level impression data
        asset_query = f"""
            SELECT
                ad_group_ad_asset_view.resource_name,
                ad_group_ad_asset_view.field_type,
                ad_group_ad_asset_view.performance_label,
                asset.text_asset.text,
                campaign.name,
                ad_group.id,
                ad_group.name,
                ad_group_ad.ad.id,
                metrics.impressions
            FROM ad_group_ad_asset_view
            WHERE campaign.id = {campaign_id}
              AND campaign.status != 'REMOVED'
              AND ad_group.status != 'REMOVED'
              AND ad_group_ad.status != 'REMOVED'
              AND asset.type = 'TEXT'
              AND {date_range}
        """
        
        asset_response = ga_service.search(customer_id=st.session_state.selected_account['id'], query=asset_query)
        
        # Aggregate asset performance
        asset_data = {}
        
        for row in asset_response:
            asset_view = row.ad_group_ad_asset_view
            asset = row.asset
            campaign = row.campaign
            ad_group = row.ad_group
            ad = row.ad_group_ad.ad
            metrics = row.metrics
            
            asset_text = asset.text_asset.text
            asset_type = 'Headline' if asset_view.field_type.name == 'HEADLINE' else 'Description'
            performance_label = asset_view.performance_label.name.replace('_', ' ').title()
            
            ad_key = f"{ad_group.id}_{ad.id}"
            asset_key = f"{asset_text}_{asset_type}_{ad_key}"
            
            if asset_key not in asset_data:
                asset_data[asset_key] = {
                    'campaign_name': campaign.name,
                    'ad_group_id': str(ad_group.id),
                    'ad_group_name': ad_group.name,
                    'ad_id': str(ad.id),
                    'asset_text': asset_text,
                    'asset_type': asset_type,
                    'performance_label': performance_label,
                    'asset_impressions': 0,
                    'ad_key': ad_key
                }
            
            asset_data[asset_key]['asset_impressions'] += metrics.impressions or 0
        
        # Step 3: Calculate impression share and attribute ad performance to assets
        final_assets = []
        
        # Group assets by ad to calculate impression shares
        assets_by_ad = {}
        for asset_key, asset_info in asset_data.items():
            ad_key = asset_info['ad_key']
            if ad_key not in assets_by_ad:
                assets_by_ad[ad_key] = []
            assets_by_ad[ad_key].append((asset_key, asset_info))
        
        # For each ad, calculate asset impression shares and attribute performance
        for ad_key, assets_in_ad in assets_by_ad.items():
            if ad_key not in ad_performance:
                continue  # Skip if we don't have ad performance data
            
            ad_perf = ad_performance[ad_key]
            
            # Calculate total asset impressions for this ad
            total_asset_impressions = sum(asset_info['asset_impressions'] for _, asset_info in assets_in_ad)
            
            if total_asset_impressions == 0:
                continue  # Skip if no asset impressions
            
            # Attribute ad performance to each asset based on impression share
            for asset_key, asset_info in assets_in_ad:
                impression_share = asset_info['asset_impressions'] / total_asset_impressions
                
                # Create attributed asset performance
                attributed_asset = {
                    'campaign_name': asset_info['campaign_name'],
                    'ad_group_id': asset_info['ad_group_id'],
                    'ad_group_name': asset_info['ad_group_name'],
                    'ad_id': asset_info['ad_id'],
                    'ad_name': ad_perf['ad_name'],
                    'asset_text': asset_info['asset_text'],
                    'asset_type': asset_info['asset_type'],
                    'performance_label': asset_info['performance_label'],
                    
                    # Asset-level data
                    'asset_impressions': asset_info['asset_impressions'],
                    'impression_share': impression_share,
                    
                    # Ad-level data
                    'ad_impressions': ad_perf['impressions'],
                    'ad_clicks': ad_perf['clicks'],
                    'ad_conversions': ad_perf['conversions'],
                    'ad_cost': ad_perf['cost'],
                    'ad_ctr': ad_perf['ctr'],
                    'ad_cpc': ad_perf['cpc'],
                    
                    # Attributed performance (asset gets credit based on impression share)
                    'attributed_clicks': ad_perf['clicks'] * impression_share,
                    'attributed_conversions': ad_perf['conversions'] * impression_share,
                    'attributed_cost': ad_perf['cost'] * impression_share,
                }
                
                # Calculate attributed CTR and CPC
                if asset_info['asset_impressions'] > 0:
                    attributed_asset['attributed_ctr'] = attributed_asset['attributed_clicks'] / asset_info['asset_impressions']
                else:
                    attributed_asset['attributed_ctr'] = 0
                
                if attributed_asset['attributed_clicks'] > 0:
                    attributed_asset['attributed_cpc'] = attributed_asset['attributed_cost'] / attributed_asset['attributed_clicks']
                else:
                    attributed_asset['attributed_cpc'] = 0
                
                final_assets.append(attributed_asset)
        
        return final_assets
        
    except GoogleAdsException as ex:
        st.error(f"❌ Error getting campaign assets: {ex}")
        return []

def display_performance_analysis():
    """Display clean asset performance analysis with actual asset-level data."""
    if not st.session_state.assets_data:
        return False
    
    st.header("📊 Asset Performance Analysis")
    st.caption(f"Campaign: {st.session_state.selected_campaign['name']}")
    
    assets_df = pd.DataFrame(st.session_state.assets_data)
    
    if assets_df.empty:
        st.warning("⚠️ No asset performance data found")
        return False
    
    # Quick stats
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        total_assets = len(assets_df)
        st.metric("Total Assets", total_assets)
    
    with col2:
        total_impressions = assets_df['asset_impressions'].sum()
        st.metric("Asset Impressions", f"{total_impressions:,}")
    
    with col3:
        total_conversions = assets_df['attributed_conversions'].sum()
        st.metric("Attributed Conversions", f"{total_conversions:.1f}")
    
    with col4:
        avg_ctr = assets_df['attributed_ctr'].mean()
        st.metric("Avg Attributed CTR", f"{avg_ctr:.2%}")
    
    with col5:
        best_count = len(assets_df[assets_df['performance_label'] == 'Best'])
        st.metric("'Best' Assets", best_count)
    
    with col6:
        good_count = len(assets_df[assets_df['performance_label'] == 'Good'])
        st.metric("'Good' Assets", good_count)
    
    # Filter controls
    st.subheader("🔍 Filters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        asset_type_filter = st.selectbox("Asset Type", ["All", "Headline", "Description"])
    
    with col2:
        performance_filter = st.selectbox("Google Label", ["All", "Best", "Good", "Learning", "Low", "Pending"])
    
    with col3:
        min_impressions = st.number_input("Min Asset Impressions", min_value=0, value=100, step=50)
    
    # Apply filters
    filtered_df = assets_df.copy()
    
    if asset_type_filter != "All":
        filtered_df = filtered_df[filtered_df['asset_type'] == asset_type_filter]
    
    if performance_filter != "All":
        filtered_df = filtered_df[filtered_df['performance_label'] == performance_filter]
    
    filtered_df = filtered_df[filtered_df['asset_impressions'] >= min_impressions]
    
    # Sort by attributed conversions (highest first), then by asset impressions
    filtered_df = filtered_df.sort_values(['attributed_conversions', 'asset_impressions'], ascending=[False, False])
    
    st.subheader(f"📋 Assets ({len(filtered_df)} shown)")
    
    if filtered_df.empty:
        st.info("No assets match the current filters")
        return True
    
    # Add view options
    view_option = st.radio("View:", ["Table View", "Hierarchical View"], horizontal=True)
    
    if view_option == "Hierarchical View":
        # Group by Ad Group and Ad for hierarchical display
        st.subheader("🏗️ Hierarchical Structure")
        
        for ad_group_name in filtered_df['ad_group_name'].unique():
            st.markdown(f"### 📁 Ad Group: {ad_group_name}")
            
            ad_group_assets = filtered_df[filtered_df['ad_group_name'] == ad_group_name]
            
            for ad_id in ad_group_assets['ad_id'].unique():
                ad_assets = ad_group_assets[ad_group_assets['ad_id'] == ad_id]
                ad_name = ad_assets.iloc[0]['ad_name']
                
                total_ad_conversions = ad_assets.iloc[0]['ad_conversions']
                total_ad_impressions = ad_assets.iloc[0]['ad_impressions']
                total_ad_cost = ad_assets.iloc[0]['ad_cost']
                
                st.markdown(f"#### 📄 Ad: {ad_name} (ID: {ad_id})")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Ad Impressions", f"{total_ad_impressions:,}")
                with col2:
                    st.metric("Ad Conversions", f"{total_ad_conversions:.1f}")
                with col3:
                    st.metric("Ad Cost", f"${total_ad_cost:.2f}")
                with col4:
                    if total_ad_cost > 0 and total_ad_conversions > 0:
                        cpa = total_ad_cost / total_ad_conversions
                        st.metric("Ad CPA", f"${cpa:.2f}")
                
                # Show assets for this ad
                st.markdown("**Assets in this Ad:**")
                
                for _, asset in ad_assets.iterrows():
                    with st.expander(f"[{asset['asset_type'][0]}] {asset['asset_text'][:50]}... ({asset['performance_label']})"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**Full Text:** {asset['asset_text']}")
                            st.write(f"**Asset Impressions:** {asset['asset_impressions']:,}")
                            st.write(f"**Impression Share:** {asset['impression_share']:.1%}")
                        
                        with col2:
                            st.write(f"**Attributed Clicks:** {asset['attributed_clicks']:.1f}")
                            st.write(f"**Attributed Conversions:** {asset['attributed_conversions']:.2f}")
                            st.write(f"**Attributed CTR:** {asset['attributed_ctr']:.2%}")
                            st.write(f"**Attributed Cost:** ${asset['attributed_cost']:.2f}")
                
                st.markdown("---")
        
        return True
    
    # Create a comprehensive table with hierarchical context
    display_df = filtered_df.copy()
    
    # Format data for display
    display_df['Asset'] = display_df.apply(lambda x: f"[{x['asset_type'][0]}] {x['asset_text'][:50]}{'...' if len(x['asset_text']) > 50 else ''}", axis=1)
    display_df['Label'] = display_df['performance_label'].apply(lambda x: 
        '🟢 Best' if x == 'Best' else
        '🔵 Good' if x == 'Good' else  
        '🟡 Learning' if x == 'Learning' else
        '🔴 Low' if x == 'Low' else
        '⚪ Pending' if x == 'Pending' else
        '⚫ Unknown'
    )
    # Asset-level metrics
    display_df['Asset Impr'] = display_df['asset_impressions'].apply(lambda x: f"{x:,}")
    display_df['Impr Share'] = display_df['impression_share'].apply(lambda x: f"{x:.1%}")
    
    # Attributed metrics (based on impression share)
    display_df['Attr Clicks'] = display_df['attributed_clicks'].apply(lambda x: f"{x:.1f}")
    display_df['Attr Conv'] = display_df['attributed_conversions'].apply(lambda x: f"{x:.2f}")
    display_df['Attr CTR'] = display_df['attributed_ctr'].apply(lambda x: f"{x:.2%}")
    display_df['Attr Cost'] = display_df['attributed_cost'].apply(lambda x: f"${x:.2f}")
    
    # Context
    display_df['Ad Group'] = display_df['ad_group_name']
    display_df['Ad ID'] = display_df['ad_id']
    
    # Calculate Asset Effectiveness Score
    # Pre-calculate percentiles to avoid index issues
    filtered_df_reset = filtered_df.reset_index(drop=True)
    ctr_percentiles = filtered_df_reset['attributed_ctr'].rank(pct=True) * 100
    conv_percentiles = filtered_df_reset['attributed_conversions'].rank(pct=True) * 100
    max_impression_share = filtered_df_reset['impression_share'].max()
    
    def calculate_effectiveness_score(row):
        """Calculate a composite effectiveness score (0-100) based on multiple factors."""
        score = 0
        
        # Google performance label (50% of score) - increased weight and scores
        label_scores = {
            'Best': 100,
            'Good': 85,  # Increased from 75 to 85
            'Learning': 40,  # Decreased from 50 to 40
            'Pending': 20,   # Decreased from 25 to 20
            'Low': 0
        }
        score += label_scores.get(row['performance_label'], 20) * 0.5  # Increased from 0.4 to 0.5
        
        # Impression share (30% of score) - higher is better
        if max_impression_share > 0:
            impression_score = (row['impression_share'] / max_impression_share) * 100
            score += impression_score * 0.3
        
        # Use pre-calculated percentiles with proper index
        idx = row.name
        if idx < len(ctr_percentiles):
            score += ctr_percentiles.iloc[idx] * 0.1  # Attributed CTR percentile (10% of score)
            score += conv_percentiles.iloc[idx] * 0.1  # Attributed conversions percentile (10% of score)
        
        return min(100, max(0, score))  # Ensure score is between 0-100
    
    # Add effectiveness score using the reset dataframe with error handling
    try:
        effectiveness_scores = []
        for idx, row in filtered_df_reset.iterrows():
            try:
                score = calculate_effectiveness_score(row)
                effectiveness_scores.append(score)
            except Exception as e:
                st.warning(f"Error calculating effectiveness score for asset {idx}: {e}")
                effectiveness_scores.append(0)  # Default to 0 if error
        
        display_df['Effectiveness Score'] = effectiveness_scores
    except Exception as e:
        st.error(f"Error calculating effectiveness scores: {e}")
        # Fallback: create default scores
        display_df['Effectiveness Score'] = [50.0] * len(display_df)
    
    # Add effectiveness score threshold slider
    st.subheader("🎯 Asset Selection Controls")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        effectiveness_threshold = st.slider(
            "Asset Effectiveness Threshold (%)",
            min_value=0,
            max_value=100,
            value=50,  # Default to 50% - assets below this are selected
            step=5,
            help="Assets with effectiveness scores below this threshold will be automatically selected for replacement"
        )
    
    with col2:
        st.metric("Assets Below Threshold", len(display_df[display_df['Effectiveness Score'] < effectiveness_threshold]))
    
    # Auto-select assets based on threshold
    display_df['Recommended'] = display_df['Effectiveness Score'] < effectiveness_threshold
    
    # Select final columns including effectiveness score
    final_columns = ['Recommended', 'Asset', 'Label', 'Effectiveness Score', 'Asset Impr', 'Impr Share', 'Attr Clicks', 'Attr Conv', 'Attr CTR', 'Attr Cost', 'Ad Group', 'Ad ID']
    
    # Style the table
    def highlight_performance(row):
        if 'Best' in row['Label']:
            return ['background-color: #d4edda'] * len(row)
        elif 'Good' in row['Label']:
            return ['background-color: #d1ecf1'] * len(row)
        elif 'Low' in row['Label']:
            return ['background-color: #f8d7da'] * len(row)
        else:
            return [''] * len(row)
    
    styled_df = display_df[final_columns].style.apply(highlight_performance, axis=1)
    
    # Display editable dataframe with checkboxes
    edited_df = st.data_editor(
        display_df[final_columns], 
        use_container_width=True, 
        height=500,
        column_config={
            "Recommended": st.column_config.CheckboxColumn(
                "Recommend for Replacement",
                help="Check to include this asset in replacement recommendations",
                default=False,
            ),
            "Effectiveness Score": st.column_config.NumberColumn(
                "Effectiveness Score (%)",
                help="Composite score based on Google label, impression share, CTR, and conversions",
                format="%.1f%%"
            )
        },
        disabled=["Asset", "Label", "Effectiveness Score", "Asset Impr", "Impr Share", "Attr Clicks", "Attr Conv", "Attr CTR", "Attr Cost", "Ad Group", "Ad ID"]
    )
    
    # Store the edited selection in session state
    if not edited_df.empty:
        # Reset indices to ensure alignment and add unique identifiers
        filtered_df_reset = filtered_df.reset_index(drop=True)
        edited_df_reset = edited_df.reset_index(drop=True)
        
        # Map back to original data with selection using proper indexing
        selected_for_replacement = []
        for idx, row in edited_df_reset.iterrows():
            if row['Recommended'] and idx < len(filtered_df_reset):
                original_asset = filtered_df_reset.iloc[idx].copy()
                
                # Get effectiveness score from the display_df using iloc for safe access
                effectiveness_score = 0
                if idx < len(display_df):
                    try:
                        effectiveness_score = display_df.iloc[idx]['Effectiveness Score']
                    except (KeyError, IndexError):
                        effectiveness_score = 0
                
                # Ensure we have all the necessary fields for replacement generation
                asset_dict = {
                    'asset_text': original_asset.get('asset_text', ''),
                    'asset_type': original_asset.get('asset_type', ''),
                    'performance_label': original_asset.get('performance_label', ''),
                    'asset_impressions': original_asset.get('asset_impressions', 0),
                    'attributed_ctr': original_asset.get('attributed_ctr', 0),
                    'attributed_conversions': original_asset.get('attributed_conversions', 0),
                    'attributed_cost': original_asset.get('attributed_cost', 0),
                    'ad_group_name': original_asset.get('ad_group_name', ''),
                    'ad_group_id': original_asset.get('ad_group_id', ''),
                    'ad_id': original_asset.get('ad_id', ''),
                    'ad_name': original_asset.get('ad_name', ''),
                    'impression_share': original_asset.get('impression_share', 0),
                    'effectiveness_score': effectiveness_score
                }
                selected_for_replacement.append(asset_dict)
        
        st.session_state.manually_selected_assets = selected_for_replacement
        
        # Show selection summary
        if selected_for_replacement:
            st.info(f"✅ {len(selected_for_replacement)} assets selected for replacement")
    
    # Asset detail view (collapsible and collapsed by default)
    if len(filtered_df) > 0:
        with st.expander("🔍 Asset Details", expanded=False):
            # Create simple asset selector
            asset_options = []
            for idx, row in filtered_df.iterrows():
                short_text = row['asset_text'][:50] + "..." if len(row['asset_text']) > 50 else row['asset_text']
                asset_options.append(f"{row['asset_type']}: {short_text}")
            
            selected_asset_idx = st.selectbox("Select asset for details:", range(len(asset_options)), format_func=lambda x: asset_options[x])
            
            if selected_asset_idx is not None:
                selected_asset = filtered_df.iloc[selected_asset_idx]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Asset Information:**")
                    st.write(f"**Text:** {selected_asset['asset_text']}")
                    st.write(f"**Type:** {selected_asset['asset_type']}")
                    st.write(f"**Ad Group:** {selected_asset['ad_group_name']}")
                    st.write(f"**Ad ID:** {selected_asset['ad_id']} ({selected_asset['ad_name']})")
                    st.write(f"**Google Label:** {selected_asset['performance_label']}")
                
                with col2:
                    st.write("**Asset Performance:**")
                    st.write(f"**Asset Impressions:** {selected_asset['asset_impressions']:,}")
                    st.write(f"**Impression Share:** {selected_asset['impression_share']:.1%}")
                    st.write("**Attributed Performance:**")
                    st.write(f"**Clicks:** {selected_asset['attributed_clicks']:.1f}")
                    st.write(f"**Conversions:** {selected_asset['attributed_conversions']:.2f}")
                    st.write(f"**CTR:** {selected_asset['attributed_ctr']:.2%}")
                    st.write(f"**Cost:** ${selected_asset['attributed_cost']:.2f}")
    
    # Analysis trigger
    st.subheader("🎯 AI Analysis")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write("**Generate insights from high-performing assets**")
        st.caption("Analyzes assets with high impressions and good Google labels")
    
    with col2:
        if st.button("🧠 Generate Creative Insights", type="primary", use_container_width=True):
            # Filter for analysis: high attributed conversions + good labels (exclude pending)
            analysis_assets = filtered_df[
                (filtered_df['attributed_conversions'] >= filtered_df['attributed_conversions'].quantile(0.3)) &  # Top 70% by attributed conversions
                (filtered_df['performance_label'].isin(['Best', 'Good', 'Learning', 'Low'])) &  # Exclude Pending
                (filtered_df['performance_label'] != 'Pending')
            ]
            
            if len(analysis_assets) == 0:
                st.warning("No high-performing assets found. Try adjusting filters.")
            else:
                generate_creative_insights(analysis_assets)
    
    # Display creative insights if they've been generated
    if st.session_state.get('insights_generated', False):
        display_creative_insights_section()
    
    return True
    if 'large_dataset_confirmed' in st.session_state:
        del st.session_state.large_dataset_confirmed
    
    # Now actually run the analysis
    generate_creative_analysis(assets_df)

def generate_creative_insights(assets_df):
    """Generate creative insights only (Step 1)."""
    
    if not st.session_state.openai_api_key:
        st.error("❌ OpenAI API key not found. Please check your environment variables.")
        return
    
    # Filter assets with impressions and exclude 'Pending' performance labels
    assets_with_impressions = assets_df[
        (assets_df['asset_impressions'] > 0) & 
        (assets_df['performance_label'] != 'Pending')
    ].copy()
    
    if assets_with_impressions.empty:
        st.error("❌ No assets with impressions found (excluding 'Pending' labels).")
        return
    
    with st.spinner("🤖 Analyzing creative patterns..."):
        try:
            generator = CreativeGenerator(api_key=st.session_state.openai_api_key)
            
            # Prepare data for LLM - convert to format expected by generator
            # Map new field names to old ones expected by creative_generator
            assets_with_impressions['asset'] = assets_with_impressions['asset_text']
            assets_with_impressions['performance_category'] = assets_with_impressions['performance_label']
            
            # Map attributed metrics to expected field names
            assets_with_impressions['impressions'] = assets_with_impressions['asset_impressions']
            assets_with_impressions['clicks'] = assets_with_impressions['attributed_clicks']
            assets_with_impressions['ctr'] = assets_with_impressions['attributed_ctr']
            assets_with_impressions['conversions'] = assets_with_impressions['attributed_conversions']
            assets_with_impressions['cost'] = assets_with_impressions['attributed_cost']
            
            # Add other commonly expected fields
            if 'ad_group' not in assets_with_impressions.columns:
                assets_with_impressions['ad_group'] = assets_with_impressions['ad_group_name']
            
            # Debug: Print available columns
            logger.info(f"Available columns for creative generator: {list(assets_with_impressions.columns)}")
            logger.info(f"Assets for analysis: {len(assets_with_impressions)} total")
            logger.info(f"Performance label distribution: {assets_with_impressions['performance_category'].value_counts().to_dict()}")
            
            # Ensure no NaN values that could cause issues
            assets_with_impressions = assets_with_impressions.fillna(0)
            
            # Step 1: Classify asset types (categorize headlines with performance labels)
            assets_with_types, classification_summary = generator.classify_asset_types(assets_with_impressions)
            
            # Step 2: Generate insights from the categorized assets
            logger.info("Generating creative insights from categorized assets...")
            insights = generator.generate_creative_insights(assets_with_types)
            logger.info(f"Generated insights keys: {list(insights.keys()) if insights else 'None'}")
            
            # Use campaign context for fallbacks
            campaign_name = st.session_state.selected_campaign.get('name', 'Campaign')
            fallback_audience = f"Users interested in {campaign_name.lower().replace('health', '').replace('-', ' ').strip()}"
            fallback_context = f"Campaign: {campaign_name}"
            
            # Store insights and processed data in session state
            st.session_state.creative_insights = insights
            st.session_state.processed_assets = assets_with_types
            st.session_state.suggested_target_audience = insights.get('suggested_target_audience', fallback_audience)
            st.session_state.suggested_brief = insights.get('suggested_brief', fallback_context)
            st.session_state.insights_generated = True
            
            st.success("✅ Creative insights generated!")
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error generating creative insights: {e}")
            logger.error(f"Error in generate_creative_insights: {e}")

def identify_assets_for_replacement(target_audience, additional_context, selected_ad_groups=None):
    """Identify assets that need replacement based on performance criteria (Step 1)."""
    
    if not st.session_state.get('processed_assets') is not None:
        st.error("❌ No processed assets found. Please generate insights first.")
        return
    
    if not st.session_state.get('creative_insights'):
        st.error("❌ No creative insights found. Please generate insights first.")
        return
    
    with st.spinner("🔍 Identifying assets that need replacement..."):
        try:
            # Prepare data grouped by ad group and ad
            assets_df = pd.DataFrame(st.session_state.processed_assets)
            
            # Filter by selected ad groups if specified
            if selected_ad_groups:
                assets_df = assets_df[assets_df['ad_group_name'].isin(selected_ad_groups)]
            
            # Calculate campaign-wide performance thresholds for comparison
            campaign_impression_threshold = assets_df['asset_impressions'].quantile(0.4)  # Bottom 40%
            campaign_ctr_threshold = assets_df['attributed_ctr'].quantile(0.4)  # Bottom 40% 
            campaign_conversion_threshold = assets_df['attributed_conversions'].quantile(0.4)  # Bottom 40%
            
            campaign_avg_impressions = assets_df['asset_impressions'].mean()
            campaign_avg_ctr = assets_df['attributed_ctr'].mean()
            campaign_avg_conversions = assets_df['attributed_conversions'].mean()
            
            logger.info(f"Campaign thresholds - Impressions: {campaign_impression_threshold:.0f}, CTR: {campaign_ctr_threshold:.2%}, Conversions: {campaign_conversion_threshold:.1f}")
            
            # Group assets by ad group and ad for context, but use campaign-wide criteria
            assets_for_replacement = []
            
            for ad_group_name in assets_df['ad_group_name'].unique():
                ad_group_assets = assets_df[assets_df['ad_group_name'] == ad_group_name]
                
                for ad_id in ad_group_assets['ad_id'].unique():
                    ad_assets = ad_group_assets[ad_group_assets['ad_id'] == ad_id]
                    ad_name = ad_assets.iloc[0]['ad_name'] if not ad_assets.empty else f"Ad {ad_id}"
                    
                    # Start with 'Low' performance labels - these are definitely underperforming
                    google_underperformers = ad_assets[
                        ad_assets['performance_category'] == 'Low'
                    ]
                    
                    # Include Learning/Pending assets ONLY with very low impression share (below 3%)
                    low_impression_share_assets = ad_assets[
                        (ad_assets['performance_category'].isin(['Learning', 'Pending'])) &
                        (ad_assets['impression_share'] < 0.03)  # Below 3% impression share
                    ]
                    
                    # Add assets that are significantly underperforming vs campaign averages
                    campaign_underperformers = ad_assets[
                        (ad_assets['asset_impressions'] <= campaign_impression_threshold) |
                        (ad_assets['attributed_ctr'] <= campaign_ctr_threshold) |
                        (ad_assets['attributed_conversions'] <= campaign_conversion_threshold)
                    ]
                    
                    # Combine all sets (union)
                    assets_needing_replacement = pd.concat([
                        google_underperformers, 
                        low_impression_share_assets,
                        campaign_underperformers
                    ]).drop_duplicates()
                    
                    # NEVER include 'Best' assets - they're explicitly high performers
                    # NEVER include 'Good' assets unless they're truly terrible on metrics
                    assets_needing_replacement = assets_needing_replacement[
                        ~(assets_needing_replacement['performance_category'] == 'Best')
                    ]
                    
                    # Only include 'Good' assets if they're REALLY bad (bottom 10% on multiple metrics)
                    good_assets_mask = assets_needing_replacement['performance_category'] == 'Good'
                    if good_assets_mask.any():
                        really_bad_goods = assets_needing_replacement[good_assets_mask]
                        bottom_10_impression = assets_df['asset_impressions'].quantile(0.1)
                        bottom_10_ctr = assets_df['attributed_ctr'].quantile(0.1) 
                        bottom_10_conversions = assets_df['attributed_conversions'].quantile(0.1)
                        
                        # Only keep 'Good' assets if they're in bottom 10% on at least 2 metrics
                        really_bad_mask = (
                            (really_bad_goods['asset_impressions'] <= bottom_10_impression).astype(int) +
                            (really_bad_goods['attributed_ctr'] <= bottom_10_ctr).astype(int) +
                            (really_bad_goods['attributed_conversions'] <= bottom_10_conversions).astype(int)
                        ) >= 2
                        
                        # Remove all 'Good' assets, then add back only the really bad ones
                        assets_needing_replacement = assets_needing_replacement[~good_assets_mask]
                        if really_bad_mask.any():
                            assets_needing_replacement = pd.concat([
                                assets_needing_replacement, 
                                really_bad_goods[really_bad_mask]
                            ])
                    
                    if not assets_needing_replacement.empty:
                        logger.info(f"Found {len(assets_needing_replacement)} assets needing replacement in ad {ad_id}")
                        
                        # Add replacement reasons for each asset
                        for _, asset in assets_needing_replacement.iterrows():
                            reasons = []
                            
                            # Primary reason: Google labels
                            if asset['performance_category'] == 'Low':
                                reasons.append(f"Google labeled as '{asset['performance_category']}'")
                            elif asset['performance_category'] in ['Learning', 'Pending'] and asset['impression_share'] < 0.03:
                                reasons.append(f"Google labeled as '{asset['performance_category']}' with very low impression share ({asset['impression_share']:.1%})")
                            
                            # Secondary reasons: low impression share (for non-Google label cases)
                            elif asset['impression_share'] < 0.03:
                                reasons.append(f"Very low impression share ({asset['impression_share']:.1%})")
                            
                            # Tertiary reasons: campaign-wide performance comparison
                            if asset['asset_impressions'] <= campaign_impression_threshold:
                                reasons.append(f"Low impressions ({asset['asset_impressions']:.0f} vs campaign avg {campaign_avg_impressions:.0f})")
                            if asset['attributed_ctr'] <= campaign_ctr_threshold:
                                reasons.append(f"Low CTR ({asset['attributed_ctr']:.2%} vs campaign avg {campaign_avg_ctr:.2%})")
                            if asset['attributed_conversions'] <= campaign_conversion_threshold:
                                reasons.append(f"Low conversions ({asset['attributed_conversions']:.1f} vs campaign avg {campaign_avg_conversions:.1f})")
                            
                            # Special case for 'Good' assets that made it through
                            if asset['performance_category'] == 'Good':
                                reasons.append("EXCEPTION: 'Good' asset but bottom 10% on multiple metrics")
                            
                            asset_info = asset.to_dict()
                            asset_info['replacement_reasons'] = '; '.join(reasons) if reasons else "Campaign underperformer"
                            assets_for_replacement.append(asset_info)
                            
                            logger.info(f"  - {asset['asset_type']}: {asset['asset'][:50]}... (Label: {asset['performance_category']}, Reasons: {'; '.join(reasons)})")
                    else:
                        logger.info(f"No assets needing replacement found in ad {ad_id}")
            
            # Store identified assets in session state
            st.session_state.assets_for_replacement = assets_for_replacement
            st.session_state.replacement_target_audience = target_audience
            st.session_state.replacement_additional_context = additional_context
            
            st.success(f"✅ Identified {len(assets_for_replacement)} assets recommended for replacement!")
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error identifying assets for replacement: {e}")
            logger.error(f"Error in identify_assets_for_replacement: {e}")

def generate_replacement_creatives():
    """Generate actual replacement recommendations for identified assets (Step 2)."""
    
    # Use manually selected assets instead of the old automatic identification
    if not st.session_state.get('manually_selected_assets'):
        st.error("❌ No assets selected for replacement. Please select assets using the checkboxes.")
        return
    
    manually_selected = st.session_state.manually_selected_assets
    target_audience = st.session_state.get('replacement_target_audience', '')
    additional_context = st.session_state.get('replacement_additional_context', '')
    
    # Convert manually selected assets to the expected format
    assets_for_replacement = []
    for asset in manually_selected:
        asset_info = {
            'asset': asset.get('asset_text', ''),
            'asset_type': asset.get('asset_type', ''),
            'performance_category': asset.get('performance_label', ''),
            'asset_impressions': asset.get('asset_impressions', 0),
            'attributed_ctr': asset.get('attributed_ctr', 0),
            'attributed_conversions': asset.get('attributed_conversions', 0),
            'attributed_cost': asset.get('attributed_cost', 0),
            'ad_group_name': asset.get('ad_group_name', ''),
            'ad_id': asset.get('ad_id', ''),
            'ad_name': asset.get('ad_name', ''),
            'impression_share': asset.get('impression_share', 0),
            'effectiveness_score': asset.get('effectiveness_score', 0)
        }
        assets_for_replacement.append(asset_info)
    
    with st.spinner("🔄 Generating replacement recommendations..."):
        try:
            # Group assets by ad for processing
            assets_by_ad = {}
            for asset in assets_for_replacement:
                ad_key = f"{asset['ad_group_name']}_{asset['ad_id']}"
                if ad_key not in assets_by_ad:
                    assets_by_ad[ad_key] = {
                        'ad_group_name': asset['ad_group_name'],
                        'ad_id': asset['ad_id'],
                        'ad_name': asset['ad_name'],
                        'assets': []
                    }
                assets_by_ad[ad_key]['assets'].append(asset)
            
            replacement_recommendations = []
            
            for ad_key, ad_data in assets_by_ad.items():
                # Get good performers from the same ad group for reference
                assets_df = pd.DataFrame(st.session_state.processed_assets)
                ad_group_assets = assets_df[assets_df['ad_group_name'] == ad_data['ad_group_name']]
                good_performers = ad_group_assets[
                    ad_group_assets['performance_category'].isin(['Best', 'Good'])
                ]
                
                # Convert assets to the format expected by the replacement function
                assets_to_replace_list = []
                for asset in ad_data['assets']:
                    assets_to_replace_list.append({
                        'text': asset['asset'],
                        'type': asset['asset_type'],
                        'performance_label': asset['performance_category'],
                        'attributed_conversions': asset['attributed_conversions'],
                        'attributed_ctr': asset['attributed_ctr']
                    })
                
                good_performers_list = []
                for _, asset in good_performers.head(10).iterrows():  # Limit to top 10 for context
                    good_performers_list.append({
                        'text': asset['asset'],
                        'type': asset['asset_type'],
                        'performance_label': asset['performance_category'],
                        'attributed_conversions': asset['attributed_conversions'],
                        'attributed_ctr': asset['attributed_ctr']
                    })
                
                replacement_request = generate_asset_replacements(
                    CreativeGenerator(api_key=st.session_state.openai_api_key),
                    assets_to_replace_list,
                    good_performers_list,
                    ad_data['ad_group_name'],
                    ad_data['ad_id'],
                    ad_data['ad_name'],
                    target_audience,
                    additional_context,
                    st.session_state.creative_insights
                )
                
                if replacement_request:
                    replacement_recommendations.extend(replacement_request)
            
            # Store results in session state
            st.session_state.replacement_recommendations = replacement_recommendations
            
            st.success(f"✅ Generated {len(replacement_recommendations)} replacement recommendations!")
            
        except Exception as e:
            st.error(f"❌ Error generating replacement recommendations: {e}")
            logger.error(f"Error in generate_replacement_creatives: {e}")

def generate_replacements_for_ad(assets_to_replace, good_performers, ad_group_name, ad_id, ad_name, target_audience, additional_context, insights):
    """Generate specific replacement recommendations for underperforming assets in an ad."""
    
    try:
        generator = CreativeGenerator(api_key=st.session_state.openai_api_key)
        
        # Prepare context
        assets_to_replace_list = []
        for _, asset in assets_to_replace.iterrows():
            assets_to_replace_list.append({
                'text': asset['asset'],
                'type': asset['asset_type'],
                'performance_label': asset['performance_category'],
                'attributed_conversions': asset['attributed_conversions'],
                'attributed_ctr': asset['attributed_ctr']
            })
        
        good_performers_list = []
        for _, asset in good_performers.head(10).iterrows():  # Limit to top 10 for context
            good_performers_list.append({
                'text': asset['asset'],
                'type': asset['asset_type'],
                'performance_label': asset['performance_category'],
                'attributed_conversions': asset['attributed_conversions'],
                'attributed_ctr': asset['attributed_ctr']
            })
        
        # Create replacement recommendations
        replacement_data = generate_asset_replacements(
            generator,
            assets_to_replace_list,
            good_performers_list,
            ad_group_name,
            ad_id,
            ad_name,
            target_audience,
            additional_context,
            insights
        )
        
        return replacement_data
        
    except Exception as e:
        logger.error(f"Error generating replacements for ad {ad_id}: {e}")
        return []

def generate_asset_replacements(generator, assets_to_replace, good_performers, ad_group_name, ad_id, ad_name, target_audience, additional_context, insights):
    """Use OpenAI to generate specific asset replacements."""
    
    logger.info(f"Generating replacements for {len(assets_to_replace)} assets in ad {ad_id}")
    
    if not assets_to_replace:
        logger.warning("No assets to replace provided")
        return []
    
    prompt = f"""
    Generate specific replacement recommendations for underperforming Google Ads assets.
    
    **CONTEXT:**
    Ad Group: {ad_group_name}
    Ad ID: {ad_id}
    Ad Name: {ad_name}
    Target Audience: {target_audience}
    Additional Context: {additional_context}
    
    **UNDERPERFORMING ASSETS TO REPLACE:**
    {chr(10).join([f"• {asset['type']}: '{asset['text']}' (Label: {asset['performance_label']}, Conv: {asset['attributed_conversions']:.2f}, CTR: {asset['attributed_ctr']:.2%})" for asset in assets_to_replace])}
    
    **HIGH-PERFORMING REFERENCE ASSETS IN AD GROUP:**
    {chr(10).join([f"• {asset['type']}: '{asset['text']}' (Label: {asset['performance_label']}, Conv: {asset['attributed_conversions']:.2f}, CTR: {asset['attributed_ctr']:.2%})" for asset in good_performers[:8]])}
    
    **CREATIVE INSIGHTS TO APPLY:**
    Key Insights: {insights.get('key_insights', 'Apply winning patterns')}
    Winning Value Propositions: {', '.join(insights.get('winning_creative_types', {}).get('value_propositions', []))}
    Winning CTAs: {', '.join(insights.get('winning_creative_types', {}).get('cta_types', []))}
    
    **REQUIREMENTS:**
    1. Generate ONE replacement for each underperforming asset
    2. Keep the same asset type (headline for headline, description for description)
    3. Maintain character limits (30 chars for headlines, 90 for descriptions)
    4. Apply winning patterns from high-performers and insights
    5. Provide specific reason for replacement
    6. Estimate expected performance improvement
    
    Return as JSON object with this exact format:
    {{
        "replacements": [
            {{
                "ad_group_name": "{ad_group_name}",
                "ad_id": "{ad_id}",
                "ad_name": "{ad_name}",
                "current_asset": "exact current text",
                "replacement_asset": "new replacement text",
                "asset_type": "Headline or Description",
                "current_performance_label": "current label",
                "reason_for_replacement": "why this asset needs replacement",
                "replacement_strategy": "what approach was used for replacement",
                "expected_improvement": "what improvement is expected"
            }}
        ]
    }}
    
    Return only valid JSON, no additional text.
    """
    
    try:
        logger.info(f"Making OpenAI API call for ad {ad_id}...")
        logger.info(f"Prompt length: {len(prompt)} characters")
        
        # Test the OpenAI client first with a simple call
        try:
            test_response = generator.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Reply with just the word 'test'"}],
                max_tokens=10
            )
            logger.info(f"Test call successful: {test_response.choices[0].message.content}")
        except Exception as e:
            logger.error(f"Test call failed: {e}")
            return []
        
        response = generator.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert Google Ads optimizer who creates targeted asset replacements based on performance data and insights. ONLY recommend replacements for clearly underperforming assets."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        logger.info("Received response from OpenAI")
        
        import json
        response_content = response.choices[0].message.content
        
        logger.info(f"Response received with {len(response_content) if response_content else 0} characters")
        
        if not response_content or response_content.strip() == "":
            logger.error("Empty response from OpenAI")
            logger.error(f"Full response: {response}")
            logger.error(f"Response choices: {response.choices}")
            logger.error(f"Message content: '{response.choices[0].message.content}'" if response.choices else "No choices")
            return []
        
        logger.info(f"Response content preview: {response_content[:200]}...")
        
        try:
            result = json.loads(response_content)
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}, Content: {response_content}")
            return []
        
        # Handle both array and object responses
        if isinstance(result, dict):
            if 'replacements' in result:
                replacements = result['replacements']
                # Ensure it's a list
                if isinstance(replacements, list):
                    return replacements
                elif isinstance(replacements, dict):
                    return [replacements]  # Single replacement as dict
                else:
                    logger.warning(f"Unexpected replacements format: {replacements}")
                    return []
            else:
                # Single replacement object
                return [result]
        elif isinstance(result, list):
            return result
        else:
            logger.warning(f"Unexpected response format: {result}")
            return []
            
    except Exception as e:
        logger.error(f"Error calling OpenAI for replacements: {e}")
        
        # Try a simpler fallback approach
        logger.info("Attempting simpler fallback prompt...")
        try:
            simple_prompt = f"""Create 1 replacement for this underperforming Google Ads asset:

Asset: "{assets_to_replace[0]['text']}"
Type: {assets_to_replace[0]['type']}
Current Performance: {assets_to_replace[0]['performance_label']}

Target Audience: {target_audience}

Return as JSON:
{{
    "replacements": [{{
        "current_asset": "{assets_to_replace[0]['text']}",
        "replacement_asset": "new improved version",
        "asset_type": "{assets_to_replace[0]['type']}",
        "reason_for_replacement": "reason",
        "expected_improvement": "expected improvement"
    }}]
}}"""

            simple_response = generator.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a Google Ads copywriter. Return only valid JSON."},
                    {"role": "user", "content": simple_prompt}
                ],
                response_format={"type": "json_object"},
                max_tokens=500
            )
            
            simple_content = simple_response.choices[0].message.content
            if simple_content:
                logger.info("Fallback approach succeeded")
                result = json.loads(simple_content)
                if 'replacements' in result and isinstance(result['replacements'], list):
                    return result['replacements']
            
        except Exception as fallback_error:
            logger.error(f"Fallback approach also failed: {fallback_error}")
        
        return []

def display_creative_insights_section():
    """Display creative insights with editable fields for target audience and brief."""
    if not st.session_state.get('creative_insights'):
        return
    
    st.subheader("🧠 AI Creative Insights")
    
    insights = st.session_state.creative_insights
    
    # Display Performance Patterns
    st.subheader("📊 Performance Patterns")
    patterns = insights.get('performance_patterns', [])
    if patterns:
        for pattern in patterns:
            st.write(f"• {pattern}")
    else:
        key_insights = insights.get('key_insights', '')
        if key_insights:
            st.write(key_insights)
        else:
            st.write("No specific patterns identified")
    
    # Display Creative Types Identified
    st.subheader("🔍 Creative Types Identified")
    creative_types = insights.get('creative_types', [])
    if creative_types:
        for i, creative_type in enumerate(creative_types, 1):
            type_name = creative_type.get('type', 'Unknown') if isinstance(creative_type, dict) else str(creative_type)
            description = creative_type.get('description', 'No description') if isinstance(creative_type, dict) else ''
            if description:
                st.write(f"{i}. **{type_name}** - {description}")
            else:
                st.write(f"{i}. **{type_name}**")
    else:
        # Show winning creative types if available
        winning_types = insights.get('winning_creative_types', {})
        if winning_types:
            col1, col2 = st.columns(2)
            with col1:
                if winning_types.get('value_propositions'):
                    st.write("**Top Value Propositions:**")
                    for vp in winning_types['value_propositions']:
                        st.write(f"• {vp}")
                
                if winning_types.get('emotional_triggers'):
                    st.write("**Effective Emotional Triggers:**")
                    for et in winning_types['emotional_triggers']:
                        st.write(f"• {et}")
            
            with col2:
                if winning_types.get('cta_types'):
                    st.write("**Effective CTAs:**")
                    for cta in winning_types['cta_types']:
                        st.write(f"• {cta}")
                
                if winning_types.get('messaging_styles'):
                    st.write("**Effective Messaging Styles:**")
                    for ms in winning_types['messaging_styles']:
                        st.write(f"• {ms}")
        else:
            st.write("No specific creative types identified")
    
    # Key insights
    if insights.get('key_insights') and not patterns:
        st.subheader("💡 Key Insights")
        st.write(insights.get('key_insights'))
    
    # Creative recommendations
    if insights.get('creative_recommendations'):
        st.subheader("📝 Creative Recommendations")
        st.write(insights.get('creative_recommendations'))
    
    st.markdown("---")
    
    # Editable Target Audience and Brief Section
    st.subheader("🎯 Creative Brief (Editable)")
    st.caption("Review and edit the target audience and context before generating new creatives")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Target Audience:**")
        target_audience = st.text_area(
            "Target Audience",
            value=st.session_state.get('suggested_target_audience', ''),
            height=100,
            help="Describe who the ads should target",
            label_visibility="collapsed"
        )
    
    with col2:
        st.write("**Context & Brief:**")
        additional_context = st.text_area(
            "Context & Brief",
            value=st.session_state.get('suggested_brief', ''),
            height=100,
            help="Additional context about the business, campaign, or creative direction",
            label_visibility="collapsed"
        )
    
    # Step 1: Show manually selected assets
    st.subheader("🎯 Step 1: Selected Assets for Replacement")
    st.caption("Assets selected using the checkboxes in the Asset Performance Analysis above")
    
    selected_assets = st.session_state.get('manually_selected_assets', [])
    
    if not selected_assets:
        st.info("📝 No assets selected yet. Please go back to the Asset Performance Analysis table and check the boxes for assets you want to replace.")
        return
    
    # Convert selected assets to the format expected by the replacement generation
    assets_for_replacement = []
    for asset in selected_assets:
        asset_info = {
            'asset': asset.get('asset_text', ''),
            'asset_type': asset.get('asset_type', ''),
            'performance_category': asset.get('performance_label', ''),
            'asset_impressions': asset.get('asset_impressions', 0),
            'attributed_ctr': asset.get('attributed_ctr', 0),
            'attributed_conversions': asset.get('attributed_conversions', 0),
            'ad_group_name': asset.get('ad_group_name', ''),
            'ad_id': asset.get('ad_id', ''),
            'ad_name': asset.get('ad_name', ''),
            'impression_share': asset.get('impression_share', 0),
            'replacement_reasons': f"Manually selected for replacement"  # Simple reason since user selected
        }
        assets_for_replacement.append(asset_info)
    
    # Store in session state for compatibility with existing code
    st.session_state.assets_for_replacement = assets_for_replacement
    
    # Step 2: Show selected assets and generate replacements
    if assets_for_replacement:
        st.markdown("---")
        st.subheader("📋 Assets Recommended for Replacement")
        
        assets_for_replacement = st.session_state.assets_for_replacement
        
        # Create display table
        display_data = []
        for asset in assets_for_replacement:
            display_data.append({
                'Asset Type': asset['asset_type'],
                'Asset Text': asset['asset'][:60] + '...' if len(asset['asset']) > 60 else asset['asset'],
                'Performance Label': asset['performance_category'],
                'Asset Impressions': f"{asset['asset_impressions']:,}",
                'Attributed CTR': f"{asset['attributed_ctr']:.2%}",
                'Attributed Conversions': f"{asset['attributed_conversions']:.1f}",
                'Replacement Reasons': asset['replacement_reasons'],
                'Ad Group': asset['ad_group_name'],
                'Ad ID': asset['ad_id']
            })
        
        # Display the table
        df_display = pd.DataFrame(display_data)
        st.dataframe(df_display, use_container_width=True, hide_index=True)
        
        # Show summary
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Assets", len(assets_for_replacement))
        with col2:
            headline_count = len([a for a in assets_for_replacement if a['asset_type'] == 'Headline'])
            st.metric("Headlines", headline_count)
        with col3:
            description_count = len([a for a in assets_for_replacement if a['asset_type'] == 'Description'])
            st.metric("Descriptions", description_count)
        with col4:
            ad_groups_count = len(set(a['ad_group_name'] for a in assets_for_replacement))
            st.metric("Ad Groups", ad_groups_count)
        
        # Step 2: Generate Replacements
        st.subheader("🔄 Step 2: Generate Replacement Creatives")
        st.caption("Now generate specific replacement recommendations for the identified assets")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"**Ready to generate {len(assets_for_replacement)} replacement recommendations?**")
            st.caption("This will use AI to create specific replacement suggestions for each underperforming asset")
        
        with col2:
            if st.button("🚀 Generate Replacements", type="primary", use_container_width=True):
                generate_replacement_creatives()
    
    # Display replacement recommendations in the same view
    if st.session_state.get('replacement_recommendations'):
        st.markdown("---")
        display_creative_insights()

def display_creative_insights():
    """Display replacement recommendations."""
    if not st.session_state.get('replacement_recommendations'):
        return
    
    st.header("🔄 Asset Replacement Recommendations")
    
    recommendations = st.session_state.replacement_recommendations
    
    if not recommendations:
        st.info("No replacement recommendations generated yet.")
        return
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Replacements", len(recommendations))
    
    with col2:
        ad_groups_count = len(set(r['ad_group_name'] for r in recommendations))
        st.metric("Ad Groups", ad_groups_count)
    
    with col3:
        ads_count = len(set(r['ad_id'] for r in recommendations))
        st.metric("Ads Affected", ads_count)
    
    with col4:
        headline_count = len([r for r in recommendations if r.get('asset_type') == 'Headline'])
        st.metric("Headlines", headline_count)
    
    # Create a clean table sorted by Ad Group → Ad → Asset
    st.subheader("📋 Replacement Recommendations")
    
    # Prepare data for table with requested column order and simplifications
    table_data = []
    for rec in recommendations:
        # Simplify Asset Type to just H or D
        asset_type_short = 'H' if rec.get('asset_type', '') == 'Headline' else 'D' if rec.get('asset_type', '') == 'Description' else rec.get('asset_type', '')
        
        table_data.append({
            'Current Asset': rec.get('current_asset', ''),
            'Replacement Asset': rec.get('replacement_asset', ''),
            'Reason': rec.get('reason_for_replacement', ''),
            'Ad Group': rec.get('ad_group_name', ''),
            'Ad ID': rec.get('ad_id', ''),
            'Type': asset_type_short,
            'Current Label': rec.get('current_performance_label', '')
        })
    
    # Create DataFrame and sort
    df = pd.DataFrame(table_data)
    if not df.empty:
        # Sort by Ad Group → Ad ID → Asset Type
        df = df.sort_values(['Ad Group', 'Ad ID', 'Type'])
        
        # Display the table with requested column order
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                'Current Asset': st.column_config.TextColumn('Current Asset', width='large'),
                'Replacement Asset': st.column_config.TextColumn('Replacement Asset', width='large'),
                'Reason': st.column_config.TextColumn('Reason', width='medium'),
                'Ad Group': st.column_config.TextColumn('Ad Group', width='medium'),
                'Ad ID': st.column_config.TextColumn('Ad ID', width='small'),
                'Type': st.column_config.TextColumn('Type', width='small'),
                'Current Label': st.column_config.TextColumn('Label', width='small')
            }
        )
    
    # Export options
    st.subheader("📤 Export Recommendations")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Export as CSV", use_container_width=True):
            export_replacements_csv(recommendations)
    
    with col2:
        if st.button("📝 Export as Text", use_container_width=True):
            export_replacements_text(recommendations)
    
    with col3:
        if st.button("📊 Export Full Report", use_container_width=True):
            if st.session_state.get('creative_insights'):
                export_replacements_full_report(st.session_state.creative_insights, recommendations)
            else:
                st.error("No insights available for full report")

def export_replacements_csv(recommendations):
    """Export replacement recommendations as CSV."""
    
    # Create structured dataframe
    export_data = []
    for rec in recommendations:
        export_data.append({
            'Ad Group': rec.get('ad_group_name', ''),
            'Ad ID': rec.get('ad_id', ''),
            'Ad Name': rec.get('ad_name', ''),
            'Asset Type': rec.get('asset_type', ''),
            'Current Asset': rec.get('current_asset', ''),
            'Replacement Asset': rec.get('replacement_asset', ''),
            'Current Performance': rec.get('current_performance_label', ''),
            'Reason for Replacement': rec.get('reason_for_replacement', ''),
            'Replacement Strategy': rec.get('replacement_strategy', ''),
            'Expected Improvement': rec.get('expected_improvement', '')
        })
    
    export_df = pd.DataFrame(export_data)
    csv = export_df.to_csv(index=False)
    
    st.download_button(
        label="📥 Download Replacements CSV",
        data=csv,
        file_name=f"asset_replacements_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

def export_replacements_text(recommendations):
    """Export replacement recommendations as text."""
    content = []
    content.append("GOOGLE ADS ASSET REPLACEMENT RECOMMENDATIONS")
    content.append("=" * 50)
    content.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    content.append("")
    
    # Group by ad group
    from collections import defaultdict
    grouped_recs = defaultdict(list)
    for rec in recommendations:
        grouped_recs[rec['ad_group_name']].append(rec)
    
    for ad_group_name, ad_group_recs in grouped_recs.items():
        content.append(f"AD GROUP: {ad_group_name}")
        content.append("-" * len(f"AD GROUP: {ad_group_name}"))
        content.append("")
        
        # Group by ad within ad group
        ad_groups = defaultdict(list)
        for rec in ad_group_recs:
            ad_groups[rec['ad_id']].append(rec)
        
        for ad_id, ad_recs in ad_groups.items():
            ad_name = ad_recs[0]['ad_name'] if ad_recs else f"Ad {ad_id}"
            content.append(f"  AD: {ad_name} (ID: {ad_id})")
            content.append("")
            
            for i, rec in enumerate(ad_recs, 1):
                content.append(f"    REPLACEMENT {i}:")
                content.append(f"    Type: {rec.get('asset_type', 'Unknown')}")
                content.append(f"    Current: {rec.get('current_asset', '')}")
                content.append(f"    Replace with: {rec.get('replacement_asset', '')}")
                content.append(f"    Current Performance: {rec.get('current_performance_label', '')}")
                content.append(f"    Reason: {rec.get('reason_for_replacement', '')}")
                content.append(f"    Expected Improvement: {rec.get('expected_improvement', '')}")
                content.append("")
        
        content.append("")
    
    text_content = "\n".join(content)
    
    st.download_button(
        label="📥 Download Text Report",
        data=text_content,
        file_name=f"asset_replacements_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain"
    )

def export_replacements_full_report(insights, recommendations):
    """Export full replacement analysis report."""
    content = []
    content.append("GOOGLE ADS REPLACEMENT ANALYSIS REPORT")
    content.append("=" * 50)
    content.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    content.append("")
    
    # Insights section
    content.append("AI CREATIVE INSIGHTS")
    content.append("-" * 30)
    content.append(f"Key Insights: {insights.get('key_insights', 'Not specified')}")
    content.append("")
    
    if insights.get('winning_creative_types'):
        winning_types = insights['winning_creative_types']
        if winning_types.get('value_propositions'):
            content.append("Winning Value Propositions:")
            for vp in winning_types['value_propositions']:
                content.append(f"• {vp}")
            content.append("")
        
        if winning_types.get('cta_types'):
            content.append("Winning CTA Types:")
            for cta in winning_types['cta_types']:
                content.append(f"• {cta}")
            content.append("")
    
    content.append("Performance Patterns:")
    for pattern in insights.get('performance_patterns', []):
        content.append(f"• {pattern}")
    content.append("")
    
    # Replacement recommendations summary
    content.append("REPLACEMENT RECOMMENDATIONS SUMMARY")
    content.append("-" * 40)
    content.append(f"Total Replacements: {len(recommendations)}")
    
    ad_groups_count = len(set(r['ad_group_name'] for r in recommendations))
    content.append(f"Ad Groups Affected: {ad_groups_count}")
    
    ads_count = len(set(r['ad_id'] for r in recommendations))
    content.append(f"Ads Affected: {ads_count}")
    content.append("")
    
    # Detailed recommendations
    content.append("DETAILED REPLACEMENT RECOMMENDATIONS")
    content.append("-" * 40)
    
    # Group by ad group
    from collections import defaultdict
    grouped_recs = defaultdict(list)
    for rec in recommendations:
        grouped_recs[rec['ad_group_name']].append(rec)
    
    for ad_group_name, ad_group_recs in grouped_recs.items():
        content.append(f"AD GROUP: {ad_group_name}")
        content.append("-" * len(f"AD GROUP: {ad_group_name}"))
        content.append("")
        
        # Group by ad within ad group
        ad_groups = defaultdict(list)
        for rec in ad_group_recs:
            ad_groups[rec['ad_id']].append(rec)
        
        for ad_id, ad_recs in ad_groups.items():
            ad_name = ad_recs[0]['ad_name'] if ad_recs else f"Ad {ad_id}"
            content.append(f"  AD: {ad_name} (ID: {ad_id})")
            content.append("")
            
            for i, rec in enumerate(ad_recs, 1):
                content.append(f"    REPLACEMENT {i} ({rec.get('asset_type', 'Unknown')}):")
                content.append(f"    Current: {rec.get('current_asset', '')}")
                content.append(f"    Replace with: {rec.get('replacement_asset', '')}")
                content.append(f"    Reason: {rec.get('reason_for_replacement', '')}")
                content.append(f"    Strategy: {rec.get('replacement_strategy', '')}")
                content.append(f"    Expected Improvement: {rec.get('expected_improvement', '')}")
                content.append("")
        
        content.append("")
    
    report_content = "\n".join(content)
    
    st.download_button(
        label="📥 Download Full Report",
        data=report_content,
        file_name=f"replacement_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain"
    )

def display_navigation():
    """Display navigation breadcrumb and back/next buttons."""
    
    pages = {
        'account_selection': '🏢 Account Selection',
        'campaign_selection': '📊 Campaign Selection', 
        'performance_analysis': '📈 Performance Analysis',
        'creative_insights': '🧠 Creative Insights'
    }
    
    current_page = st.session_state.current_page
    
    # Breadcrumb navigation
    st.markdown("### 📍 Navigation")
    breadcrumb = []
    for page_key, page_name in pages.items():
        if page_key == current_page:
            breadcrumb.append(f"**{page_name}**")
        elif (page_key == 'account_selection' or 
              (page_key == 'campaign_selection' and st.session_state.selected_account) or
              (page_key == 'performance_analysis' and st.session_state.selected_campaign) or
              (page_key == 'creative_insights' and st.session_state.assets_data)):
            breadcrumb.append(f"[{page_name}]")
        else:
            breadcrumb.append(page_name)
    
    st.markdown(" → ".join(breadcrumb))
    
    # Navigation buttons
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if current_page != 'account_selection':
            if st.button("⬅️ Back", key="nav_back"):
                if current_page == 'campaign_selection':
                    st.session_state.current_page = 'account_selection'
                elif current_page == 'performance_analysis':
                    st.session_state.current_page = 'campaign_selection'
                elif current_page == 'creative_insights':
                    st.session_state.current_page = 'performance_analysis'
                st.rerun()
    
    with col3:
        # Show next button if conditions are met
        if current_page == 'account_selection' and st.session_state.selected_account:
            if st.button("Next ➡️", key="nav_next_1"):
                st.session_state.current_page = 'campaign_selection'
                st.rerun()
        elif current_page == 'campaign_selection' and st.session_state.assets_data:
            if st.button("Next ➡️", key="nav_next_2"):
                st.session_state.current_page = 'performance_analysis'
                st.rerun()
        elif current_page == 'performance_analysis' and st.session_state.creative_insights:
            if st.button("Next ➡️", key="nav_next_3"):
                st.session_state.current_page = 'creative_insights'
                st.rerun()
    
    st.markdown("---")

def check_password():
    """Returns `True` if the user had the correct password."""
    
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == "ap123":
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True

def main():
    """Main application function."""
    # Password protection
    if not check_password():
        st.title("🎯 Google Ads Creative Generator")
        st.markdown("Please enter the password to access the application.")
        return
    
    # Initialize session state
    initialize_session_state()
    
    # Main app header
    st.title("🎯 Google Ads Creative Generator")
    st.markdown("**AI-powered creative optimization using Google Ads performance data**")
    
    # Display navigation
    display_navigation()
    
    # Display current page content
    current_page = st.session_state.current_page
    
    if current_page == 'account_selection':
        display_account_selection()
    
    elif current_page == 'campaign_selection':
        if st.session_state.selected_account:
            display_campaign_selection()
        else:
            st.error("❌ No account selected. Please go back and select an account.")
            if st.button("⬅️ Go to Account Selection"):
                st.session_state.current_page = 'account_selection'
                st.rerun()
    
    elif current_page == 'performance_analysis':
        if st.session_state.assets_data:
            display_performance_analysis()
        else:
            st.error("❌ No campaign data loaded. Please go back and select a campaign.")
            if st.button("⬅️ Go to Campaign Selection"):
                st.session_state.current_page = 'campaign_selection'
                st.rerun()
    
    elif current_page == 'creative_insights':
        if st.session_state.creative_insights:
            display_creative_insights()
        else:
            st.error("❌ No creative insights generated. Please go back and generate analysis.")
            if st.button("⬅️ Go to Performance Analysis"):
                st.session_state.current_page = 'performance_analysis'
                st.rerun()
    
    # Sidebar with current state summary
    with st.sidebar:
        st.header("📋 Current State")
        
        if st.session_state.selected_account:
            st.success(f"✅ **Account:** {st.session_state.selected_account['name'][:30]}...")
        else:
            st.info("⏳ No account selected")
        
        if st.session_state.selected_campaign:
            st.success(f"✅ **Campaign:** {st.session_state.selected_campaign['name'][:30]}...")
        else:
            st.info("⏳ No campaign selected")
        
        if st.session_state.assets_data:
            st.success(f"✅ **Assets:** {len(st.session_state.assets_data)} loaded")
        else:
            st.info("⏳ No assets loaded")
        
        if st.session_state.creative_insights:
            st.success("✅ **AI Analysis:** Complete")
        else:
            st.info("⏳ No AI analysis")
        
        # Reset button
        st.markdown("---")
        if st.button("🔄 Start Over", type="secondary"):
            # Reset all state
            st.session_state.current_page = 'account_selection'
            st.session_state.selected_account = None
            st.session_state.campaigns_data = None
            st.session_state.selected_campaign = None
            st.session_state.assets_data = None
            st.session_state.creative_insights = None
            st.session_state.generated_creative = None
            st.rerun()

if __name__ == "__main__":
    main() 