---
title: Google Ads Creative Generator
emoji: 🎯
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.28.0
app_file: app.py
pinned: false
---

# 🎯 Google Ads Creative Generator

AI-powered creative generation and analysis tool for Google Ads campaigns.

## Features

- **📊 Campaign Analysis**: Load and analyze campaign asset performance data
- **🧠 Creative Insights**: AI-powered classification and performance analysis
- **🎨 Creative Generation**: Generate new headlines and descriptions based on winning patterns
- **📥 Export Functionality**: Download generated creative assets

## How to Use

1. **Upload Campaign Data**: Add your Google Ads asset performance CSV files to the `data/` directory
2. **Select Campaign**: Choose a campaign to analyze from the dropdown
3. **Analyze Performance**: Review asset performance by category (Best, Good, Learning, Low)
4. **Get AI Insights**: Use AI to classify creative types and identify winning patterns
5. **Generate New Creative**: Create new headlines and descriptions based on insights
6. **Export Results**: Download your generated creative assets

## Setup

1. Enter your OpenAI API key in the sidebar
2. Upload your Google Ads asset performance CSV files
3. Start analyzing and generating!

## Data Format

The app expects Google Ads Asset Performance Report CSV files with columns:
- Asset
- Asset type (Headline/Description)
- Performance categories (Best, Good, Low, Learning, Unrated)
- Performance metrics (Impressions, Clicks, CTR, Conversions, etc.)

---

*Built with Streamlit and OpenAI GPT-4* 