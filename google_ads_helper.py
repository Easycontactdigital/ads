#!/usr/bin/env python3
"""
Google Ads API Helper Functions
Easily load configuration and create authenticated clients
"""

import json
from google.ads.googleads.client import GoogleAdsClient

def load_config(config_file="google_ads_config.json"):
    """Load Google Ads configuration from JSON file"""
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ Configuration file {config_file} not found")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in {config_file}: {e}")
        return None

def create_client(config=None, target_customer_id=None):
    """
    Create authenticated Google Ads client
    
    Args:
        config: Configuration dict (if None, loads from file)
        target_customer_id: Override the target customer ID
    
    Returns:
        GoogleAdsClient instance
    """
    if config is None:
        config = load_config()
        if not config:
            return None
    
    # Build client configuration
    client_config = {
        "developer_token": config["authentication"]["developer_token"],
        "refresh_token": config["authentication"]["refresh_token"],
        "client_id": config["authentication"]["client_id"],
        "client_secret": config["authentication"]["client_secret"],
        "login_customer_id": config["working_configuration"]["login_customer_id"],
        "use_proto_plus": config["authentication"]["use_proto_plus"]
    }
    
    try:
        client = GoogleAdsClient.load_from_dict(client_config)
        print(f"✅ Google Ads client created successfully")
        print(f"   Login Customer ID: {client_config['login_customer_id']}")
        
        # Use target customer ID if provided, otherwise use from config
        customer_id = target_customer_id or config["working_configuration"]["target_customer_id"]
        print(f"   Target Customer ID: {customer_id}")
        
        return client, customer_id
        
    except Exception as e:
        print(f"❌ Failed to create Google Ads client: {e}")
        return None, None

def test_connection(client=None, customer_id=None):
    """Test the Google Ads API connection"""
    if client is None:
        client, customer_id = create_client()
        if not client:
            return False
    
    try:
        print(f"🔍 Testing connection to customer {customer_id}...")
        
        ga_service = client.get_service("GoogleAdsService")
        query = """
            SELECT 
                customer.descriptive_name,
                customer.currency_code
            FROM customer
            LIMIT 1
        """
        
        response = ga_service.search(customer_id=customer_id, query=query)
        customer_data = list(response)
        
        if customer_data:
            customer = customer_data[0].customer
            print(f"✅ Connection successful!")
            print(f"   Account: {customer.descriptive_name}")
            print(f"   Currency: {customer.currency_code}")
            return True
        else:
            print("❌ No customer data returned")
            return False
            
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        return False

def update_config_with_client_account(client_account_id, client_account_name):
    """Update the configuration file with the discovered client account"""
    config = load_config()
    if not config:
        return False
    
    # Update the target customer ID to the client account
    config["working_configuration"]["target_customer_id"] = client_account_id
    config["working_configuration"]["client_account_name"] = client_account_name
    config["working_configuration"]["note"] = f"Updated to use client account {client_account_name} which has actual campaign data"
    
    # Save back to file
    try:
        with open("google_ads_config.json", 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✅ Configuration updated with client account: {client_account_name} ({client_account_id})")
        return True
    except Exception as e:
        print(f"❌ Failed to update configuration: {e}")
        return False

if __name__ == "__main__":
    # Test the helper functions
    print("🧪 Testing Google Ads Helper Functions")
    print("=" * 50)
    
    # Load config
    config = load_config()
    if config:
        print("✅ Configuration loaded successfully")
        
        # Create client
        client, customer_id = create_client(config)
        if client:
            # Test connection
            test_connection(client, customer_id)
    else:
        print("❌ Failed to load configuration") 