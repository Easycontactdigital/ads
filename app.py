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
    
    # Select final columns
    final_columns = ['Asset', 'Label', 'Asset Impr', 'Impr Share', 'Attr Clicks', 'Attr Conv', 'Attr CTR', 'Attr Cost', 'Ad Group', 'Ad ID']
    
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
    
    st.dataframe(styled_df, use_container_width=True, height=500)
    
    # Asset detail view
    if len(filtered_df) > 0:
        st.subheader("🔍 Asset Details")
        
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

def generate_new_creatives(target_audience, additional_context):
    """Generate new creatives based on insights (Step 2)."""
    
    if not st.session_state.get('processed_assets') is not None:
        st.error("❌ No processed assets found. Please generate insights first.")
        return
    
    if not st.session_state.get('creative_insights'):
        st.error("❌ No creative insights found. Please generate insights first.")
        return
    
    with st.spinner("✨ Generating new creative variations..."):
        try:
            generator = CreativeGenerator(api_key=st.session_state.openai_api_key)
            
            new_creative = generator.generate_new_creatives_with_insights(
                asset_data=st.session_state.processed_assets,
                creative_insights=st.session_state.creative_insights,
                target_audience=target_audience,
                additional_context=additional_context,
                use_categories=['Best', 'Good'],
                num_headlines=30,
                num_descriptions=30
            )
            
            # Store results in session state
            st.session_state.generated_creative = new_creative
            st.session_state.current_page = 'creative_insights'
            
            st.success("✅ New creatives generated!")
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error generating new creatives: {e}")
            logger.error(f"Error in generate_new_creatives: {e}")

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
    
    # Generate Creatives Button
    st.subheader("✨ Generate New Creatives")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write("**Ready to generate new creative variations?**")
        st.caption("Will create 30 headlines and 30 descriptions based on insights above")
    
    with col2:
        if st.button("🚀 Generate Creatives", type="primary", use_container_width=True):
            if target_audience.strip() and additional_context.strip():
                generate_new_creatives(target_audience, additional_context)
            else:
                st.error("Please fill in both Target Audience and Context & Brief fields")

def display_creative_insights():
    """Display generated creatives only."""
    if not st.session_state.generated_creative:
        return
    
    st.header("✨ Generated Creative Variations")
    
    creative = st.session_state.generated_creative
    
    # Strategy summary (if available)
    if creative.get('strategy_summary'):
        st.subheader("📋 Creative Strategy")
        st.write(creative['strategy_summary'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 New Headlines")
        headlines = creative.get('headlines', [])
        for i, headline in enumerate(headlines, 1):
            st.write(f"{i}. {headline}")
    
    with col2:
        st.subheader("📄 New Descriptions")
        descriptions = creative.get('descriptions', [])
        for i, description in enumerate(descriptions, 1):
            st.write(f"{i}. {description}")
    
    # Character count info (if available)
    char_counts = creative.get('character_counts', {})
    if char_counts:
        st.subheader("📊 Character Count Analysis")
        col1, col2 = st.columns(2)
        with col1:
            avg_headline_chars = char_counts.get('headlines_avg', 'N/A')
            st.metric("Avg Headline Length", f"{avg_headline_chars:.1f} chars" if isinstance(avg_headline_chars, (int, float)) else avg_headline_chars)
        with col2:
            avg_desc_chars = char_counts.get('descriptions_avg', 'N/A')
            st.metric("Avg Description Length", f"{avg_desc_chars:.1f} chars" if isinstance(avg_desc_chars, (int, float)) else avg_desc_chars)
    
    # Export options
    st.subheader("📤 Export Creative")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Export as CSV", use_container_width=True):
            export_creative_csv(creative)
    
    with col2:
        if st.button("📝 Export as Text", use_container_width=True):
            export_creative_text(creative)
    
    with col3:
        if st.button("📊 Export Full Report", use_container_width=True):
            if st.session_state.get('creative_insights'):
                export_full_report(st.session_state.creative_insights, creative)
            else:
                st.error("No insights available for full report")

def export_creative_csv(creative):
    """Export creative variations as CSV."""
    # Create dataframes
    headlines_df = pd.DataFrame(creative.get('headlines', []), columns=['Headlines'])
    descriptions_df = pd.DataFrame(creative.get('descriptions', []), columns=['Descriptions'])
    
    # Combine into one dataframe
    max_len = max(len(headlines_df), len(descriptions_df))
    
    export_df = pd.DataFrame({
        'Headlines': headlines_df['Headlines'].tolist() + [''] * (max_len - len(headlines_df)),
        'Descriptions': descriptions_df['Descriptions'].tolist() + [''] * (max_len - len(descriptions_df))
    })
    
    # Convert to CSV
    csv = export_df.to_csv(index=False)
    
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name=f"creative_variations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

def export_creative_text(creative):
    """Export creative variations as text."""
    content = []
    content.append("GOOGLE ADS CREATIVE VARIATIONS")
    content.append("=" * 40)
    content.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    content.append("")
    
    content.append("HEADLINES:")
    content.append("-" * 20)
    for i, headline in enumerate(creative.get('headlines', []), 1):
        content.append(f"{i}. {headline}")
    
    content.append("")
    content.append("DESCRIPTIONS:")
    content.append("-" * 20)
    for i, description in enumerate(creative.get('descriptions', []), 1):
        content.append(f"{i}. {description}")
    
    text_content = "\n".join(content)
    
    st.download_button(
        label="📥 Download Text",
        data=text_content,
        file_name=f"creative_variations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain"
    )

def export_full_report(insights, creative):
    """Export full analysis report."""
    content = []
    content.append("GOOGLE ADS CREATIVE ANALYSIS REPORT")
    content.append("=" * 50)
    content.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    content.append("")
    
    # Insights section
    content.append("AI CREATIVE INSIGHTS")
    content.append("-" * 30)
    content.append(f"Target Audience: {insights.get('target_audience', 'Not specified')}")
    content.append("")
    content.append(f"Context & Brief: {insights.get('context_brief', 'Not specified')}")
    content.append("")
    
    content.append("Creative Types Identified:")
    for creative_type in insights.get('creative_types', []):
        content.append(f"• {creative_type.get('type', 'Unknown')}: {creative_type.get('description', 'No description')}")
    content.append("")
    
    content.append("Performance Patterns:")
    for pattern in insights.get('performance_patterns', []):
        content.append(f"• {pattern}")
    content.append("")
    
    # Creative variations
    content.append("GENERATED CREATIVE VARIATIONS")
    content.append("-" * 40)
    
    content.append("Headlines:")
    for i, headline in enumerate(creative.get('headlines', []), 1):
        content.append(f"{i}. {headline}")
    content.append("")
    
    content.append("Descriptions:")
    for i, description in enumerate(creative.get('descriptions', []), 1):
        content.append(f"{i}. {description}")
    
    report_content = "\n".join(content)
    
    st.download_button(
        label="📥 Download Full Report",
        data=report_content,
        file_name=f"creative_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
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
        if st.session_state["password"] == "adspassword123":
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