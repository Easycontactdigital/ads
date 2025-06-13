"""
Google Ads Creative Generator - Streamlit App
Main interface for generating new ad creatives using campaign data analysis and OpenAI
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
import os
import sys
import json
from datetime import datetime

# Add src directory to path
sys.path.append('src')

# Load environment variables
load_dotenv()

# Import our custom modules
try:
    from creative_generator import CreativeGenerator
    from csv_data_loader import (
        list_campaign_csvs,
        load_campaign_asset_data,
        get_campaign_summary
    )
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
    .creative-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .asset-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 3px solid #28a745;
    }
    .headline-item {
        background-color: #e3f2fd;
        padding: 0.5rem;
        margin: 0.25rem 0;
        border-radius: 0.25rem;
        border-left: 3px solid #2196f3;
    }
    .description-item {
        background-color: #f3e5f5;
        padding: 0.5rem;
        margin: 0.25rem 0;
        border-radius: 0.25rem;
        border-left: 3px solid #9c27b0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    if 'selected_campaign' not in st.session_state:
        st.session_state.selected_campaign = None
    if 'campaign_asset_data' not in st.session_state:
        st.session_state.campaign_asset_data = None
    if 'classified_assets' not in st.session_state:
        st.session_state.classified_assets = None
    if 'creative_insights' not in st.session_state:
        st.session_state.creative_insights = None
    if 'generated_creative' not in st.session_state:
        st.session_state.generated_creative = None
    if 'openai_api_key' not in st.session_state:
        st.session_state.openai_api_key = os.getenv('OPENAI_API_KEY', '')

def display_campaign_selection():
    """Display campaign selection interface."""
    st.header("📊 Campaign Selection")
    st.caption("Choose a campaign to analyze from your data directory")
    
    # List available campaigns
    available_campaigns = list_campaign_csvs()
    
    if not available_campaigns:
        st.error("❌ No campaign CSV files found in the 'data' directory")
        st.info("Please add your campaign CSV files to the 'data' directory and refresh the page.")
        return False
    
    # Campaign selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_campaign = st.selectbox(
            "Select Campaign",
            available_campaigns,
            index=0 if available_campaigns else None,
            help="Choose a campaign CSV file to analyze"
        )
        
        if selected_campaign != st.session_state.selected_campaign:
            st.session_state.selected_campaign = selected_campaign
            st.session_state.campaign_asset_data = None
            st.session_state.classified_assets = None
            st.session_state.creative_insights = None
            st.session_state.generated_creative = None
    
    with col2:
        if st.button("📥 Load Campaign Data", type="primary", use_container_width=True):
            if selected_campaign:
                with st.spinner(f"Loading data from {selected_campaign}..."):
                    try:
                        asset_data = load_campaign_asset_data(selected_campaign)
                        if not asset_data.empty:
                            st.session_state.campaign_asset_data = asset_data
                            st.success(f"✅ Loaded {len(asset_data)} assets from {selected_campaign}")
                            st.rerun()
                        else:
                            st.error("No asset data found in the selected campaign file")
                    except Exception as e:
                        st.error(f"Error loading campaign data: {e}")
    
    # Display campaign summary if data is loaded
    if st.session_state.campaign_asset_data is not None:
        display_campaign_overview(st.session_state.campaign_asset_data)
        return True
    
    return False

def display_campaign_overview(asset_data):
    """Display overview of loaded campaign data."""
    st.subheader("📈 Campaign Overview")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_assets = len(asset_data)
        st.markdown(f"""
        <div class="metric-card">
            <h4>📊 Total Assets</h4>
            <h2>{total_assets}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        headlines_count = len(asset_data[asset_data['asset_type'] == 'Headline'])
        st.markdown(f"""
        <div class="metric-card">
            <h4>📝 Headlines</h4>
            <h2>{headlines_count}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        descriptions_count = len(asset_data[asset_data['asset_type'] == 'Description'])
        st.markdown(f"""
        <div class="metric-card">
            <h4>📄 Descriptions</h4>
            <h2>{descriptions_count}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_ctr = asset_data['ctr'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <h4>📈 Avg CTR</h4>
            <h2>{avg_ctr:.2%}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Performance breakdown
    st.subheader("🎯 Performance Categories")
    
    category_counts = asset_data['performance_category'].value_counts()
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        best_count = category_counts.get('Best', 0)
        st.metric("🏆 Best", best_count)
    
    with col2:
        good_count = category_counts.get('Good', 0)
        st.metric("✅ Good", good_count)
    
    with col3:
        learning_count = category_counts.get('Learning', 0)
        st.metric("📚 Learning", learning_count)
    
    with col4:
        low_count = category_counts.get('Low', 0)
        st.metric("📉 Low", low_count)
    
    with col5:
        unrated_count = category_counts.get('Unrated', 0)
        st.metric("❓ Unrated", unrated_count)

def display_creative_analysis():
    """Display creative analysis interface."""
    st.header("🔍 Creative Analysis")
    st.caption("Analyze performance by category and asset type")
    
    asset_data = st.session_state.campaign_asset_data
    
    # Filter controls
    col1, col2 = st.columns(2)
    
    with col1:
        selected_categories = st.multiselect(
            "Performance Categories",
            ['Best', 'Good', 'Learning', 'Low', 'Unrated'],
            default=['Best', 'Good'],
            help="Select which performance categories to display"
        )
    
    with col2:
        selected_asset_types = st.multiselect(
            "Asset Types",
            ['Headline', 'Description'],
            default=['Headline', 'Description'],
            help="Select which asset types to display"
        )
    
    # Filter data
    filtered_data = asset_data[
        (asset_data['performance_category'].isin(selected_categories)) &
        (asset_data['asset_type'].isin(selected_asset_types))
    ]
    
    if filtered_data.empty:
        st.warning("No assets match the selected filters")
        return
    
    # Display filtered assets
    for category in selected_categories:
        category_data = filtered_data[filtered_data['performance_category'] == category]
        
        if category_data.empty:
            continue
        
        # Category header with emoji
        category_emoji = {'Best': '🏆', 'Good': '✅', 'Learning': '📚', 'Low': '📉', 'Unrated': '❓'}
        st.subheader(f"{category_emoji.get(category, '📊')} {category} Performing Assets ({len(category_data)})")
        
        # Split by asset type
        headlines = category_data[category_data['asset_type'] == 'Headline']
        descriptions = category_data[category_data['asset_type'] == 'Description']
        
        col1, col2 = st.columns(2)
        
        with col1:
            if not headlines.empty and 'Headline' in selected_asset_types:
                st.markdown("**📝 Headlines:**")
                for _, asset in headlines.head(10).iterrows():
                    st.markdown(f"""
                    <div class="asset-card">
                        <strong>{asset['asset']}</strong><br>
                        <small>Impressions: {asset['impressions']:,} | CTR: {asset['ctr']:.2%} | Conversions: {asset['conversions']:.1f}</small>
                    </div>
                    """, unsafe_allow_html=True)
        
        with col2:
            if not descriptions.empty and 'Description' in selected_asset_types:
                st.markdown("**📄 Descriptions:**")
                for _, asset in descriptions.head(10).iterrows():
                    st.markdown(f"""
                    <div class="asset-card">
                        <strong>{asset['asset']}</strong><br>
                        <small>Impressions: {asset['impressions']:,} | CTR: {asset['ctr']:.2%} | Conversions: {asset['conversions']:.1f}</small>
                    </div>
                    """, unsafe_allow_html=True)

def display_creative_insights():
    """Display creative insights interface with LLM classification."""
    st.header("🧠 Creative Insights")
    st.caption("AI-powered analysis of what creative types perform best")
    
    if not st.session_state.openai_api_key:
        st.warning("⚠️ Please enter your OpenAI API key in the sidebar to get creative insights.")
        return
    
    asset_data = st.session_state.campaign_asset_data
    
    if asset_data is None or asset_data.empty:
        st.warning("Please load campaign data first")
        return
    
    # Step 1: Classify assets using LLM
    if st.session_state.classified_assets is None:
        if st.button("🤖 Classify Creative Types", type="primary", use_container_width=True):
            with st.spinner("🤖 AI is analyzing and classifying your creative assets..."):
                try:
                    generator = CreativeGenerator(st.session_state.openai_api_key)
                    classified_assets, classification_summary = generator.classify_asset_types(asset_data)
                    st.session_state.classified_assets = classified_assets
                    st.success("✅ Asset classification complete!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error classifying assets: {e}")
                    return
    
    # Display classification results
    if st.session_state.classified_assets is not None:
        st.subheader("🏷️ Creative Type Classifications")
        
        classified_data = st.session_state.classified_assets
        
        # Show classification summary
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            value_props = classified_data['value_proposition'].value_counts()
            top_vp = value_props.index[0] if len(value_props) > 0 else "Unknown"
            st.metric("Top Value Prop", top_vp, f"{value_props.iloc[0]} assets" if len(value_props) > 0 else "")
        
        with col2:
            emotional_triggers = classified_data['emotional_trigger'].value_counts()
            top_et = emotional_triggers.index[0] if len(emotional_triggers) > 0 else "Unknown"
            st.metric("Top Emotional Trigger", top_et, f"{emotional_triggers.iloc[0]} assets" if len(emotional_triggers) > 0 else "")
        
        with col3:
            cta_types = classified_data['cta_type'].value_counts()
            top_cta = cta_types.index[0] if len(cta_types) > 0 else "Unknown"
            st.metric("Top CTA Type", top_cta, f"{cta_types.iloc[0]} assets" if len(cta_types) > 0 else "")
        
        with col4:
            messaging_styles = classified_data['messaging_style'].value_counts()
            top_style = messaging_styles.index[0] if len(messaging_styles) > 0 else "Unknown"
            st.metric("Top Messaging Style", top_style, f"{messaging_styles.iloc[0]} assets" if len(messaging_styles) > 0 else "")
        
        # Step 2: Generate insights
        if st.session_state.creative_insights is None:
            if st.button("📊 Generate Performance Insights", type="secondary", use_container_width=True):
                with st.spinner("🤖 Analyzing performance patterns..."):
                    try:
                        generator = CreativeGenerator(st.session_state.openai_api_key)
                        insights = generator.generate_creative_insights(classified_data)
                        st.session_state.creative_insights = insights
                        st.success("✅ Insights generated!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error generating insights: {e}")
                        return
        
        # Display insights
        if st.session_state.creative_insights is not None:
            insights = st.session_state.creative_insights
            
            st.subheader("💡 Key Performance Insights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### ✅ What Works")
                winning_types = insights.get('winning_creative_types', {})
                
                if winning_types.get('value_propositions'):
                    st.write("**Value Propositions:**")
                    for vp in winning_types['value_propositions']:
                        st.write(f"• {vp}")
                
                if winning_types.get('emotional_triggers'):
                    st.write("**Emotional Triggers:**")
                    for et in winning_types['emotional_triggers']:
                        st.write(f"• {et}")
                
                if winning_types.get('cta_types'):
                    st.write("**Call-to-Actions:**")
                    for cta in winning_types['cta_types']:
                        st.write(f"• {cta}")
            
            with col2:
                st.markdown("### ❌ What to Avoid")
                avoid_types = insights.get('avoid_creative_types', {})
                
                if avoid_types.get('value_propositions'):
                    st.write("**Value Propositions:**")
                    for vp in avoid_types['value_propositions']:
                        st.write(f"• {vp}")
                
                if avoid_types.get('emotional_triggers'):
                    st.write("**Emotional Triggers:**")
                    for et in avoid_types['emotional_triggers']:
                        st.write(f"• {et}")
                
                if avoid_types.get('cta_types'):
                    st.write("**Call-to-Actions:**")
                    for cta in avoid_types['cta_types']:
                        st.write(f"• {cta}")
            
            # Key insights
            st.subheader("🎯 Key Insights")
            st.info(insights.get('key_insights', 'No key insights available'))
            
            # Recommendations
            st.subheader("📝 Recommendations")
            st.success(insights.get('creative_recommendations', 'No recommendations available'))

def display_creative_generation():
    """Display creative generation interface."""
    st.header("🎨 Generate New Creative")
    st.caption("Create new headlines and descriptions based on your insights")
    
    if not st.session_state.openai_api_key:
        st.warning("⚠️ Please enter your OpenAI API key in the sidebar to generate creative.")
        return
    
    if st.session_state.classified_assets is None:
        st.warning("Please complete creative analysis and insights first")
        return
    
    if st.session_state.creative_insights is None:
        st.warning("Please generate creative insights first")
        return
    
    # Generation settings
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Get suggested values from insights if available
        suggested_audience = ""
        suggested_brief = ""
        if st.session_state.creative_insights:
            suggested_audience = st.session_state.creative_insights.get('suggested_target_audience', '')
            suggested_brief = st.session_state.creative_insights.get('suggested_brief', '')
        
        target_audience = st.text_input(
            "🎯 Target Audience",
            value=suggested_audience if suggested_audience else "Small business owners looking for van insurance",
            help="Describe your target audience (auto-filled from insights)"
        )
        
        additional_context = st.text_area(
            "📝 Additional Context/Brief",
            value=suggested_brief if suggested_brief else "Focus on competitive pricing, fast quotes, and comprehensive coverage",
            help="Any specific messaging requirements or brand guidelines (auto-filled from insights)"
        )
        
        use_categories = st.multiselect(
            "📊 Use Assets From Categories",
            ['Best', 'Good', 'Learning', 'Low'],
            default=['Best', 'Good'],
            help="Which performance categories to use as input for generation"
        )
    
    with col2:
        num_headlines = st.slider("📝 Number of Headlines", 10, 50, 30)
        num_descriptions = st.slider("📄 Number of Descriptions", 5, 30, 30)
        
        st.info(f"**Total Creative Elements:** {num_headlines + num_descriptions}\n\n"
               f"• {num_headlines} unique headlines\n"
               f"• {num_descriptions} unique descriptions")
    
    # Generate creative
    if st.button("🚀 Generate New Creative", type="primary", use_container_width=True):
        with st.spinner("🤖 Generating new creative based on your winning patterns..."):
            try:
                generator = CreativeGenerator(st.session_state.openai_api_key)
                
                generated_creative = generator.generate_new_creatives_with_insights(
                    asset_data=st.session_state.classified_assets,
                    creative_insights=st.session_state.creative_insights,
                    target_audience=target_audience,
                    additional_context=additional_context,
                    use_categories=use_categories,
                    num_headlines=num_headlines,
                    num_descriptions=num_descriptions
                )
                
                st.session_state.generated_creative = generated_creative
                st.success(f"✅ Generated {len(generated_creative.get('headlines', []))} headlines and {len(generated_creative.get('descriptions', []))} descriptions!")
                
            except Exception as e:
                st.error(f"Error generating creative: {e}")
                return
    
    # Display generated creative
    if st.session_state.generated_creative:
        creative = st.session_state.generated_creative
        
        st.subheader("🎯 Generated Creative")
        
        # Strategy summary
        if creative.get('strategy_summary'):
            st.info(f"**Strategy:** {creative['strategy_summary']}")
        
        if creative.get('pattern_usage'):
            st.success(f"**Pattern Usage:** {creative['pattern_usage']}")
        
        # Display headlines and descriptions
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📝 Headlines")
            headlines = creative.get('headlines', [])
            for i, headline in enumerate(headlines, 1):
                st.markdown(f"""
                <div class="headline-item">
                    <strong>{i}.</strong> {headline} <small>({len(headline)} chars)</small>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 📄 Descriptions")
            descriptions = creative.get('descriptions', [])
            for i, description in enumerate(descriptions, 1):
                st.markdown(f"""
                <div class="description-item">
                    <strong>{i}.</strong> {description} <small>({len(description)} chars)</small>
                </div>
                """, unsafe_allow_html=True)
        
        # Character count summary
        if creative.get('character_counts'):
            char_counts = creative['character_counts']
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Avg Headline Length", f"{char_counts.get('headlines_avg', 0):.1f} chars")
            with col2:
                st.metric("Avg Description Length", f"{char_counts.get('descriptions_avg', 0):.1f} chars")
        
        # Export functionality
        st.markdown("---")
        if st.button("📥 Export Creative", use_container_width=True):
            # Convert to CSV format
            export_data = []
            
            # Add headlines
            for i, headline in enumerate(headlines, 1):
                export_data.append({
                    'type': 'headline',
                    'sequence': i,
                    'text': headline,
                    'character_count': len(headline),
                    'strategy': creative.get('strategy_summary', ''),
                    'target_audience': target_audience
                })
            
            # Add descriptions
            for i, description in enumerate(descriptions, 1):
                export_data.append({
                    'type': 'description',
                    'sequence': i,
                    'text': description,
                    'character_count': len(description),
                    'strategy': creative.get('strategy_summary', ''),
                    'target_audience': target_audience
                })
            
            export_df = pd.DataFrame(export_data)
            csv = export_df.to_csv(index=False)
            
            st.download_button(
                label="📄 Download as CSV",
                data=csv,
                file_name=f"generated_creative_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

def check_password():
    """Returns True if the user has entered the correct password."""

    if "password_correct" not in st.session_state:
        # First run, show input for password
        password_input = st.text_input("🔐 Enter Password", type="password", key="password_input")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write("*Please enter the password to access this app.*")
        with col2:
            if st.button("Enter", type="primary"):
                if password_input == "adspassword123":
                    st.session_state["password_correct"] = True
                    st.rerun()
                else:
                    st.error("😞 Password incorrect")
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error
        password_input = st.text_input("🔐 Enter Password", type="password", key="password_input")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.error("😞 Password incorrect")
        with col2:
            if st.button("Enter", type="primary"):
                if password_input == "adspassword123":
                    st.session_state["password_correct"] = True
                    st.rerun()
                else:
                    st.error("😞 Password incorrect")
        return False
    else:
        # Password correct
        return True

def main():
    """Main application function."""
    st.title("🎯 Google Ads Creative Generator")
    st.markdown("AI-powered creative generation based on campaign performance data")
    
    # Check password first
    if not check_password():
        return
    
    initialize_session_state()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # OpenAI API Key
        openai_api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=st.session_state.openai_api_key,
            help="Enter your OpenAI API key"
        )
        
        if openai_api_key != st.session_state.openai_api_key:
            st.session_state.openai_api_key = openai_api_key
        
        # Reset button
        if st.button("🔄 Reset All Data"):
            for key in list(st.session_state.keys()):
                if key != 'openai_api_key':
                    del st.session_state[key]
            st.success("All data reset!")
            st.rerun()
    
    # Main workflow
    if st.session_state.selected_campaign is None:
        # Step 1: Campaign Selection
        if not display_campaign_selection():
            return
    else:
        # Show current campaign
        st.sidebar.info(f"**Current Campaign:**\n{st.session_state.selected_campaign}")
        
        if st.sidebar.button("Change Campaign"):
            st.session_state.selected_campaign = None
            st.session_state.campaign_asset_data = None
            st.session_state.classified_assets = None
            st.session_state.creative_insights = None
            st.session_state.generated_creative = None
            st.rerun()
        
        # Main tabs
        if st.session_state.campaign_asset_data is not None:
            tab1, tab2, tab3 = st.tabs(["🔍 Creative Analysis", "🧠 Creative Insights", "🎨 Generate New Creative"])
            
            with tab1:
                display_creative_analysis()
            
            with tab2:
                display_creative_insights()
            
            with tab3:
                display_creative_generation()
        else:
            display_campaign_selection()

if __name__ == "__main__":
    main() 