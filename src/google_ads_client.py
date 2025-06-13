"""
Google Ads API Client
Handles authentication and data retrieval from Google Ads API
"""

import os
import logging
from typing import List, Dict, Any, Optional
from google.ads.googleads.client import GoogleAdsClient
from google.ads.googleads.errors import GoogleAdsException
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GoogleAdsManager:
    """Manages Google Ads API interactions for ad creative analysis and data retrieval."""
    
    def __init__(self, config_path: str = "config/google_ads_api.yaml"):
        """Initialize the Google Ads client."""
        self.config_path = config_path
        self.client = None
        self.customer_id = os.getenv('GOOGLE_ADS_CUSTOMER_ID')
        
        if not self.customer_id:
            raise ValueError("GOOGLE_ADS_CUSTOMER_ID environment variable is required")
            
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the Google Ads client with configuration."""
        try:
            # Check if service account key file exists
            service_account_path = "service-account-key.json"
            
            if os.path.exists(service_account_path):
                # Use service account authentication
                logger.info("Using service account authentication")
                credentials = {
                    'use_proto_plus': True,
                    'json_key_file_path': service_account_path
                }
                
                # Add developer token if available
                developer_token = os.getenv('GOOGLE_ADS_DEVELOPER_TOKEN')
                if developer_token:
                    credentials['developer_token'] = developer_token
                else:
                    raise ValueError("GOOGLE_ADS_DEVELOPER_TOKEN is required for service account")
                
            else:
                # Fall back to OAuth authentication
                logger.info("Using OAuth authentication")
                credentials = {
                    'developer_token': os.getenv('GOOGLE_ADS_DEVELOPER_TOKEN'),
                    'client_id': os.getenv('GOOGLE_ADS_CLIENT_ID'),
                    'client_secret': os.getenv('GOOGLE_ADS_CLIENT_SECRET'),
                    'refresh_token': os.getenv('GOOGLE_ADS_REFRESH_TOKEN'),
                    'use_proto_plus': True
                }
                
                # Add login_customer_id if provided
                login_customer_id = os.getenv('GOOGLE_ADS_LOGIN_CUSTOMER_ID')
                if login_customer_id:
                    credentials['login_customer_id'] = login_customer_id
                
                # Verify all required credentials are present
                required_creds = ['developer_token', 'client_id', 'client_secret', 'refresh_token']
                missing_creds = [key for key in required_creds if not credentials[key]]
                
                if missing_creds:
                    raise ValueError(f"Missing required OAuth credentials: {missing_creds}")
            
            self.client = GoogleAdsClient.load_from_dict(credentials)
            logger.info("Google Ads client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Google Ads client: {e}")
            raise
    
    def get_existing_ads(self, limit: int = 100, campaign_ids: list = None) -> pd.DataFrame:
        """
        Retrieve existing ads from Google Ads account.
        
        Args:
            limit: Maximum number of ads to retrieve
            campaign_ids: List of campaign IDs to filter by (optional)
            
        Returns:
            DataFrame with ad information
        """
        try:
            ga_service = self.client.get_service("GoogleAdsService")
            
            # Build campaign filter if provided
            campaign_filter = ""
            if campaign_ids:
                campaign_list = ",".join(str(id) for id in campaign_ids)
                campaign_filter = f" AND campaign.id IN ({campaign_list})"
            
            query = f"""
                SELECT
                    ad_group_ad.ad.id,
                    ad_group_ad.ad.type_,
                    ad_group_ad.ad.text_ad.headline,
                    ad_group_ad.ad.text_ad.description1,
                    ad_group_ad.ad.text_ad.description2,
                    ad_group_ad.ad.expanded_text_ad.headline_part1,
                    ad_group_ad.ad.expanded_text_ad.headline_part2,
                    ad_group_ad.ad.expanded_text_ad.headline_part3,
                    ad_group_ad.ad.expanded_text_ad.description1,
                    ad_group_ad.ad.expanded_text_ad.description2,
                    ad_group_ad.ad.responsive_search_ad.headlines,
                    ad_group_ad.ad.responsive_search_ad.descriptions,
                    ad_group_ad.status,
                    ad_group.id,
                    ad_group.name,
                    campaign.id,
                    campaign.name
                FROM ad_group_ad
                WHERE ad_group_ad.status != 'REMOVED'{campaign_filter}
                LIMIT {limit}
            """
            
            search_request = self.client.get_type("SearchGoogleAdsRequest")
            search_request.customer_id = self.customer_id
            search_request.query = query
            
            results = ga_service.search(request=search_request)
            
            ads_data = []
            for row in results:
                ad = row.ad_group_ad.ad
                ad_data = {
                    'ad_id': ad.id,
                    'ad_type': ad.type_.name,
                    'status': row.ad_group_ad.status.name,
                    'ad_group_id': row.ad_group.id,
                    'ad_group_name': row.ad_group.name,
                    'campaign_id': row.campaign.id,
                    'campaign_name': row.campaign.name,
                }
                
                # Extract headlines and descriptions based on ad type
                if ad.type_.name == 'TEXT_AD':
                    ad_data.update({
                        'headline': ad.text_ad.headline,
                        'description1': ad.text_ad.description1,
                        'description2': ad.text_ad.description2,
                    })
                elif ad.type_.name == 'EXPANDED_TEXT_AD':
                    ad_data.update({
                        'headline1': ad.expanded_text_ad.headline_part1,
                        'headline2': ad.expanded_text_ad.headline_part2,
                        'headline3': ad.expanded_text_ad.headline_part3,
                        'description1': ad.expanded_text_ad.description1,
                        'description2': ad.expanded_text_ad.description2,
                    })
                elif ad.type_.name == 'RESPONSIVE_SEARCH_AD':
                    headlines = [h.text for h in ad.responsive_search_ad.headlines]
                    descriptions = [d.text for d in ad.responsive_search_ad.descriptions]
                    ad_data.update({
                        'headlines': '|'.join(headlines),
                        'descriptions': '|'.join(descriptions),
                    })
                
                ads_data.append(ad_data)
            
            df = pd.DataFrame(ads_data)
            logger.info(f"Retrieved {len(df)} ads from Google Ads")
            return df
            
        except GoogleAdsException as ex:
            logger.error(f"Google Ads API error: {ex}")
            raise
        except Exception as e:
            logger.error(f"Error retrieving ads: {e}")
            raise
    
    def get_ad_performance(self, date_range: str = "LAST_30_DAYS", limit: int = 100, campaign_ids: list = None) -> pd.DataFrame:
        """
        Retrieve ad performance metrics from Google Ads.
        
        Args:
            date_range: Date range for performance data (e.g., 'LAST_30_DAYS', 'LAST_7_DAYS')
            limit: Maximum number of records to retrieve
            campaign_ids: List of campaign IDs to filter by (optional)
            
        Returns:
            DataFrame with performance metrics
        """
        try:
            ga_service = self.client.get_service("GoogleAdsService")
            
            # Build campaign filter if provided
            campaign_filter = ""
            if campaign_ids:
                campaign_list = ",".join(str(id) for id in campaign_ids)
                campaign_filter = f" AND campaign.id IN ({campaign_list})"
            
            query = f"""
                SELECT
                    ad_group_ad.ad.id,
                    ad_group_ad.ad.type_,
                    ad_group.id,
                    ad_group.name,
                    campaign.id,
                    campaign.name,
                    metrics.impressions,
                    metrics.clicks,
                    metrics.ctr,
                    metrics.cost_micros,
                    metrics.conversions,
                    metrics.conversion_rate,
                    metrics.cost_per_conversion
                FROM ad_group_ad
                WHERE segments.date DURING {date_range}
                    AND ad_group_ad.status != 'REMOVED'
                    AND metrics.impressions > 0{campaign_filter}
                ORDER BY metrics.impressions DESC
                LIMIT {limit}
            """
            
            search_request = self.client.get_type("SearchGoogleAdsRequest")
            search_request.customer_id = self.customer_id
            search_request.query = query
            
            results = ga_service.search(request=search_request)
            
            performance_data = []
            for row in results:
                perf_data = {
                    'ad_id': row.ad_group_ad.ad.id,
                    'ad_type': row.ad_group_ad.ad.type_.name,
                    'ad_group_id': row.ad_group.id,
                    'ad_group_name': row.ad_group.name,
                    'campaign_id': row.campaign.id,
                    'campaign_name': row.campaign.name,
                    'impressions': row.metrics.impressions,
                    'clicks': row.metrics.clicks,
                    'ctr': row.metrics.ctr,
                    'cost_micros': row.metrics.cost_micros,
                    'cost': row.metrics.cost_micros / 1_000_000,  # Convert to dollars
                    'conversions': row.metrics.conversions,
                    'conversion_rate': row.metrics.conversion_rate,
                    'cost_per_conversion': row.metrics.cost_per_conversion,
                }
                performance_data.append(perf_data)
            
            df = pd.DataFrame(performance_data)
            logger.info(f"Retrieved performance data for {len(df)} ads")
            return df
            
        except GoogleAdsException as ex:
            logger.error(f"Google Ads API error: {ex}")
            raise
        except Exception as e:
            logger.error(f"Error retrieving performance data: {e}")
            raise
    
    def get_campaigns(self) -> pd.DataFrame:
        """Get list of campaigns in the account."""
        try:
            ga_service = self.client.get_service("GoogleAdsService")
            
            query = """
                SELECT
                    campaign.id,
                    campaign.name,
                    campaign.status,
                    campaign.advertising_channel_type,
                    campaign.bidding_strategy_type
                FROM campaign
                WHERE campaign.status != 'REMOVED'
                ORDER BY campaign.name
            """
            
            search_request = self.client.get_type("SearchGoogleAdsRequest")
            search_request.customer_id = self.customer_id
            search_request.query = query
            
            results = ga_service.search(request=search_request)
            
            campaigns = []
            for row in results:
                campaign_data = {
                    'campaign_id': row.campaign.id,
                    'campaign_name': row.campaign.name,
                    'status': row.campaign.status.name,
                    'channel_type': row.campaign.advertising_channel_type.name,
                    'bidding_strategy': row.campaign.bidding_strategy_type.name,
                }
                campaigns.append(campaign_data)
            
            df = pd.DataFrame(campaigns)
            logger.info(f"Retrieved {len(df)} campaigns")
            return df
            
        except GoogleAdsException as ex:
            logger.error(f"Google Ads API error: {ex}")
            raise
        except Exception as e:
            logger.error(f"Error retrieving campaigns: {e}")
            raise 