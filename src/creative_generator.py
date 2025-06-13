"""
Creative Generator using OpenAI
Analyzes existing ad performance and generates new ad creatives
"""

import os
import logging
from typing import List, Dict, Any, Optional
import pandas as pd
from openai import OpenAI
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CreativeGenerator:
    """Generates new ad creatives using OpenAI based on existing ad performance data."""
    
    def __init__(self, api_key=None):
        """Initialize the OpenAI client."""
        # Use provided API key or fall back to environment variable
        openai_key = api_key or os.getenv('OPENAI_API_KEY')
        
        if not openai_key:
            raise ValueError("OpenAI API key is required either as parameter or OPENAI_API_KEY environment variable")
        
        self.client = OpenAI(api_key=openai_key)
    
    def analyze_top_performing_ads(self, ads_df: pd.DataFrame, performance_df: pd.DataFrame, 
                                 top_n: int = 10) -> Dict[str, Any]:
        """
        Analyze top performing ads to identify successful patterns.
        
        Args:
            ads_df: DataFrame with ad creative data
            performance_df: DataFrame with performance metrics
            top_n: Number of top performing ads to analyze
            
        Returns:
            Analysis summary with patterns and insights
        """
        try:
            # Merge ads with performance data
            merged_df = pd.merge(ads_df, performance_df, on='ad_id', how='inner')
            
            if merged_df.empty:
                logger.warning("No matching ads found between creative and performance data")
                return {}
            
            # Sort by CTR and conversion rate
            merged_df['performance_score'] = (
                merged_df['ctr'] * 0.4 + 
                merged_df['conversion_rate'] * 0.6
            )
            
            top_ads = merged_df.nlargest(top_n, 'performance_score')
            
            # Extract creative elements for analysis
            creative_elements = []
            for _, ad in top_ads.iterrows():
                if ad['ad_type'] == 'RESPONSIVE_SEARCH_AD':
                    headlines = ad['headlines'].split('|') if pd.notna(ad['headlines']) else []
                    descriptions = ad['descriptions'].split('|') if pd.notna(ad['descriptions']) else []
                    creative_elements.append({
                        'headlines': headlines,
                        'descriptions': descriptions,
                        'ctr': ad['ctr'],
                        'conversion_rate': ad['conversion_rate'],
                        'campaign_name': ad['campaign_name_x']
                    })
                elif ad['ad_type'] == 'EXPANDED_TEXT_AD':
                    headlines = [ad['headline1'], ad['headline2'], ad['headline3']]
                    descriptions = [ad['description1'], ad['description2']]
                    creative_elements.append({
                        'headlines': [h for h in headlines if pd.notna(h)],
                        'descriptions': [d for d in descriptions if pd.notna(d)],
                        'ctr': ad['ctr'],
                        'conversion_rate': ad['conversion_rate'],
                        'campaign_name': ad['campaign_name_x']
                    })
            
            analysis = {
                'top_performing_ads': creative_elements,
                'avg_ctr': top_ads['ctr'].mean(),
                'avg_conversion_rate': top_ads['conversion_rate'].mean(),
                'total_analyzed': len(top_ads)
            }
            
            logger.info(f"Analyzed {len(top_ads)} top performing ads")
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing top performing ads: {e}")
            raise
    
    def generate_new_creatives(self, analysis_data: Dict[str, Any], 
                             business_context: str, num_variations: int = 5) -> List[Dict[str, Any]]:
        """
        Generate new ad creatives based on successful patterns.
        
        Args:
            analysis_data: Analysis of top performing ads
            business_context: Description of the business/product
            num_variations: Number of creative variations to generate
            
        Returns:
            List of generated ad creatives
        """
        try:
            if not analysis_data or not analysis_data.get('top_performing_ads'):
                logger.warning("No analysis data provided, generating generic creatives")
                return self._generate_generic_creatives(business_context, num_variations)
            
            # Prepare context for OpenAI
            successful_patterns = self._extract_patterns(analysis_data['top_performing_ads'])
            
            prompt = f"""
Based on the following successful Google Ads patterns and business context, generate {num_variations} new responsive search ad variations.

BUSINESS CONTEXT:
{business_context}

SUCCESSFUL PATTERNS FROM TOP PERFORMING ADS:
Average CTR: {analysis_data.get('avg_ctr', 0):.2%}
Average Conversion Rate: {analysis_data.get('avg_conversion_rate', 0):.2%}

SUCCESSFUL HEADLINES PATTERNS:
{chr(10).join(successful_patterns['headline_patterns'])}

SUCCESSFUL DESCRIPTIONS PATTERNS:
{chr(10).join(successful_patterns['description_patterns'])}

INSTRUCTIONS:
1. Create responsive search ads with 10-15 headlines and 4 descriptions each
2. Headlines should be under 30 characters
3. Descriptions should be under 90 characters
4. Incorporate successful patterns while staying relevant to the business context
5. Use compelling calls-to-action
6. Focus on value propositions that have proven successful
7. Vary the messaging approach across variations

Please return your response as a valid JSON array of objects, each with:
- "headlines": array of 10-15 headline strings
- "descriptions": array of 4 description strings
- "theme": brief description of the creative theme
- "target_audience": suggested target audience for this variation

Return only the JSON response, no additional text.
"""

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert Google Ads copywriter who creates high-converting ad creatives based on performance data analysis."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            # Parse the JSON response
            generated_creatives = json.loads(response.choices[0].message.content)
            
            logger.info(f"Generated {len(generated_creatives)} new ad creative variations")
            return generated_creatives
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse OpenAI response as JSON: {e}")
            return self._generate_fallback_creatives(business_context, num_variations)
        except Exception as e:
            logger.error(f"Error generating new creatives: {e}")
            return self._generate_fallback_creatives(business_context, num_variations)
    
    def _extract_patterns(self, top_ads: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Extract successful patterns from top performing ads."""
        headline_patterns = []
        description_patterns = []
        
        for ad in top_ads:
            headlines = ad.get('headlines', [])
            descriptions = ad.get('descriptions', [])
            
            headline_patterns.extend(headlines)
            description_patterns.extend(descriptions)
        
        # Remove duplicates and empty strings
        headline_patterns = list(set([h for h in headline_patterns if h and h.strip()]))
        description_patterns = list(set([d for d in description_patterns if d and d.strip()]))
        
        return {
            'headline_patterns': headline_patterns[:20],  # Limit to top 20
            'description_patterns': description_patterns[:15]  # Limit to top 15
        }
    
    def _generate_generic_creatives(self, business_context: str, num_variations: int) -> List[Dict[str, Any]]:
        """Generate generic creatives when no analysis data is available."""
        prompt = f"""
Create {num_variations} Google Ads responsive search ad variations for the following business:

BUSINESS CONTEXT:
{business_context}

INSTRUCTIONS:
1. Create responsive search ads with 10-15 headlines and 4 descriptions each
2. Headlines should be under 30 characters
3. Descriptions should be under 90 characters
4. Use compelling calls-to-action
5. Focus on key value propositions
6. Vary the messaging approach across variations

Please return your response as a valid JSON array with objects containing:
- "headlines": array of 10-15 headline strings
- "descriptions": array of 4 description strings
- "theme": brief description of the creative theme
- "target_audience": suggested target audience for this variation

Return only the JSON response, no additional text.
"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "You are an expert Google Ads copywriter who creates high-converting ad creatives."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=4000
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            logger.error(f"Error generating generic creatives: {e}")
            return self._generate_fallback_creatives(business_context, num_variations)
    
    def _generate_fallback_creatives(self, business_context: str, num_variations: int) -> List[Dict[str, Any]]:
        """Generate fallback creatives when OpenAI fails."""
        logger.warning("Using fallback creative generation")
        
        fallback_creatives = []
        for i in range(num_variations):
            creative = {
                "headlines": [
                    f"Quality Products - Variation {i+1}",
                    "Shop Now & Save",
                    "Limited Time Offer",
                    "Best Deals Available",
                    "Order Today",
                    "Free Shipping",
                    "Expert Service",
                    "Top Rated",
                    "Fast Delivery",
                    "Satisfaction Guaranteed"
                ],
                "descriptions": [
                    f"Discover amazing products and services. {business_context[:50]}...",
                    "Join thousands of satisfied customers. Order now!",
                    "Premium quality at unbeatable prices. Shop today.",
                    "Experience the difference. Fast shipping available."
                ],
                "theme": f"Generic Theme {i+1}",
                "target_audience": "General audience"
            }
            fallback_creatives.append(creative)
        
        return fallback_creatives
    
    def optimize_creative_for_audience(self, creative: Dict[str, Any], 
                                     audience_description: str) -> Dict[str, Any]:
        """
        Optimize a creative for a specific audience.
        
        Args:
            creative: Generated creative to optimize
            audience_description: Description of target audience
            
        Returns:
            Optimized creative
        """
        try:
            prompt = f"""
Optimize the following Google Ads creative for this specific audience:

TARGET AUDIENCE: {audience_description}

CURRENT CREATIVE:
Theme: {creative.get('theme', 'N/A')}
Headlines: {', '.join(creative.get('headlines', []))}
Descriptions: {', '.join(creative.get('descriptions', []))}

INSTRUCTIONS:
1. Adjust language, tone, and messaging to resonate with the target audience
2. Maintain headline length under 30 characters
3. Maintain description length under 90 characters
4. Keep the same number of headlines and descriptions
5. Focus on benefits most relevant to this audience

Please return the optimized creative as a valid JSON object with the same structure:
- "headlines": array of headline strings
- "descriptions": array of description strings
- "theme": updated theme description
- "target_audience": confirmed target audience

Return only the JSON response, no additional text.
"""

            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "You are an expert at tailoring ad copy for specific audiences."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.6,
                max_tokens=2000
            )
            
            optimized_creative = json.loads(response.choices[0].message.content)
            logger.info(f"Optimized creative for audience: {audience_description}")
            return optimized_creative
            
        except Exception as e:
            logger.error(f"Error optimizing creative: {e}")
            return creative  # Return original if optimization fails

    def analyze_performance(self, performance_data):
        """Analyze performance data and provide insights."""
        try:
            # Calculate key metrics
            avg_ctr = performance_data['ctr'].mean()
            avg_conversion_rate = performance_data['conversion_rate'].mean() if 'conversion_rate' in performance_data.columns else 0
            avg_cost = performance_data['cost'].mean()
            
            # Get top and bottom performers
            top_performers = performance_data.nlargest(5, 'ctr')
            bottom_performers = performance_data.nsmallest(5, 'ctr')
            
            prompt = f"""
            As a Google Ads performance analyst, analyze this campaign data and provide insights:
            
            **Overall Performance:**
            - Average CTR: {avg_ctr:.4f}
            - Average Conversion Rate: {avg_conversion_rate:.4f}
            - Average Cost: ${avg_cost:.2f}
            
            **Top 5 Performers (CTR):**
            {top_performers[['ctr', 'conversion_rate', 'cost']].to_string() if not top_performers.empty else 'No data'}
            
            **Bottom 5 Performers (CTR):**
            {bottom_performers[['ctr', 'conversion_rate', 'cost']].to_string() if not bottom_performers.empty else 'No data'}
            
            Provide actionable insights on:
            1. What's working well
            2. What needs improvement
            3. Specific recommendations for optimization
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}]
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error analyzing performance: {str(e)}"

    def analyze_performance_patterns(self, top_performers, bottom_performers):
        """Analyze what works vs what doesn't work in ad performance."""
        try:
            # Extract headlines and descriptions from top performers
            top_headlines = []
            top_descriptions = []
            top_metrics = []
            
            for idx, ad in top_performers.iterrows():
                if isinstance(ad['headlines'], list):
                    top_headlines.extend(ad['headlines'])
                else:
                    top_headlines.append(str(ad['headlines']))
                
                if isinstance(ad['descriptions'], list):
                    top_descriptions.extend(ad['descriptions'])
                else:
                    top_descriptions.append(str(ad['descriptions']))
                
                top_metrics.append(f"CTR: {ad['ctr']*100:.2f}%, Conversions: {ad['conversions']:.0f}, Cost: ${ad['cost']:.2f}")
            
            # Extract headlines and descriptions from bottom performers
            bottom_headlines = []
            bottom_descriptions = []
            bottom_metrics = []
            
            for idx, ad in bottom_performers.iterrows():
                if isinstance(ad['headlines'], list):
                    bottom_headlines.extend(ad['headlines'])
                else:
                    bottom_headlines.append(str(ad['headlines']))
                
                if isinstance(ad['descriptions'], list):
                    bottom_descriptions.extend(ad['descriptions'])
                else:
                    bottom_descriptions.append(str(ad['descriptions']))
                
                bottom_metrics.append(f"CTR: {ad['ctr']*100:.2f}%, Conversions: {ad['conversions']:.0f}, Cost: ${ad['cost']:.2f}")
            
            prompt = f"""
            As a Google Ads expert, analyze the performance patterns between top and bottom performing ads.
            
            **TOP PERFORMING ADS:**
            Headlines: {' | '.join(top_headlines[:10])}
            Descriptions: {' | '.join(top_descriptions[:5])}
            Metrics: {' | '.join(top_metrics)}
            
            **BOTTOM PERFORMING ADS:**
            Headlines: {' | '.join(bottom_headlines[:10])}
            Descriptions: {' | '.join(bottom_descriptions[:5])}
            Metrics: {' | '.join(bottom_metrics)}
            
            Please provide your analysis in a valid JSON format with this structure:
            {{
                "what_works": "Clear explanation of successful patterns, messaging, and approaches",
                "what_doesnt_work": "Clear explanation of unsuccessful patterns and what to avoid",
                "recommendations": "Specific actionable recommendations for future creatives"
            }}
            
            Focus on:
            - Language patterns and messaging that drive results
            - Emotional triggers that work vs don't work
            - Call-to-action effectiveness
            - Value proposition clarity
            - Trust signals and credibility elements
            
            Return only the JSON response, no additional text.
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            
            import json
            try:
                return json.loads(response.choices[0].message.content)
            except:
                # Fallback if JSON parsing fails
                content = response.choices[0].message.content
                return {
                    "what_works": content[:len(content)//3],
                    "what_doesnt_work": content[len(content)//3:2*len(content)//3],
                    "recommendations": content[2*len(content)//3:]
                }
            
        except Exception as e:
            return {
                "what_works": f"Error analyzing patterns: {str(e)}",
                "what_doesnt_work": "Unable to analyze poor performers",
                "recommendations": "Please try again or check your OpenAI API key"
            }

    def generate_similar_creatives(self, top_performers, target_audience, additional_context, num_variations=3):
        """Generate creatives similar to top performers."""
        try:
            # Extract patterns from top performers
            successful_headlines = []
            successful_descriptions = []
            
            for idx, ad in top_performers.iterrows():
                if isinstance(ad['headlines'], list):
                    successful_headlines.extend(ad['headlines'][:3])  # Top 3 headlines
                else:
                    successful_headlines.append(str(ad['headlines']))
                
                if isinstance(ad['descriptions'], list):
                    successful_descriptions.extend(ad['descriptions'][:2])  # Top 2 descriptions
                else:
                    successful_descriptions.append(str(ad['descriptions']))
            
            prompt = f"""
            Create {num_variations} new Google Ads responsive search ad variations based on these TOP PERFORMING ads.
            
            **SUCCESSFUL HEADLINES:**
            {chr(10).join(f"• {h}" for h in successful_headlines[:10])}
            
            **SUCCESSFUL DESCRIPTIONS:**
            {chr(10).join(f"• {d}" for d in successful_descriptions[:5])}
            
            **Target Audience:** {target_audience}
            **Additional Context:** {additional_context}
            
            **REQUIREMENTS:**
            - Create variations that follow the successful patterns but aren't identical copies
            - Each variation needs 5-10 headlines (max 30 chars each)
            - Each variation needs 2-4 descriptions (max 90 chars each)
            - Stay true to what made the originals successful
            - Keep the same tone and value propositions that work
            
            Return as JSON array:
            [
                {{
                    "headlines": ["headline1", "headline2", ...],
                    "descriptions": ["desc1", "desc2", ...],
                    "reasoning": "Why this follows successful patterns"
                }}
            ]
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}]
            )
            
            import json
            try:
                return json.loads(response.choices[0].message.content)
            except:
                # Fallback parsing
                return self._parse_creative_fallback(response.choices[0].message.content, num_variations)
            
        except Exception as e:
            return [{"headlines": [f"Error: {str(e)}"], "descriptions": ["Please check your API key"], "reasoning": "Generation failed"}]

    def generate_experimental_creatives(self, performance_insights, target_audience, additional_context, num_variations=2):
        """Generate experimental creative ideas to test new approaches."""
        try:
            insights_text = ""
            if performance_insights:
                insights_text = f"""
                **What Currently Works:** {performance_insights.get('what_works', 'N/A')}
                **What Doesn't Work:** {performance_insights.get('what_doesnt_work', 'N/A')}
                **Recommendations:** {performance_insights.get('recommendations', 'N/A')}
                """
            
            prompt = f"""
            Create {num_variations} EXPERIMENTAL Google Ads variations that test new approaches and ideas.
            
            **Performance Insights:**
            {insights_text}
            
            **Target Audience:** {target_audience}
            **Additional Context:** {additional_context}
            
            **EXPERIMENTAL GOALS:**
            - Test completely different messaging angles
            - Try new emotional triggers or value propositions
            - Experiment with different call-to-action approaches
            - Push creative boundaries while staying relevant
            - Test contrarian or bold approaches
            
            **REQUIREMENTS:**
            - Each variation needs 5-10 headlines (max 30 chars each)
            - Each variation needs 2-4 descriptions (max 90 chars each)
            - Be creative and test new hypotheses
            - Don't just copy what's already working
            
            Return as JSON array:
            [
                {{
                    "headlines": ["headline1", "headline2", ...],
                    "descriptions": ["desc1", "desc2", ...],
                    "reasoning": "Experimental hypothesis and why this approach might work"
                }}
            ]
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}]
            )
            
            import json
            try:
                return json.loads(response.choices[0].message.content)
            except:
                # Fallback parsing
                return self._parse_creative_fallback(response.choices[0].message.content, num_variations)
            
        except Exception as e:
            return [{"headlines": [f"Error: {str(e)}"], "descriptions": ["Please check your API key"], "reasoning": "Generation failed"}]

    def _parse_creative_fallback(self, content, num_variations):
        """Fallback parser when JSON parsing fails."""
        try:
            # Simple fallback - create basic structure
            creatives = []
            for i in range(num_variations):
                creatives.append({
                    "headlines": [f"New Headline {i+1}", f"Creative {i+1}", f"Test {i+1}"],
                    "descriptions": [f"Description {i+1}", f"Test description {i+1}"],
                    "reasoning": f"Fallback creative {i+1} - check content parsing"
                })
            return creatives
        except:
            return [{"headlines": ["Parsing Error"], "descriptions": ["Check response"], "reasoning": "Fallback failed"}]

    def analyze_creative_patterns(self, top_performers, bottom_performers, best_assets, learning_assets):
        """Analyze creative copy patterns to identify what works vs what doesn't."""
        try:
            # Extract creative copy from top performers
            top_headlines = []
            top_descriptions = []
            
            for idx, ad in top_performers.iterrows():
                if isinstance(ad['headlines'], list):
                    top_headlines.extend(ad['headlines'])
                else:
                    top_headlines.append(str(ad['headlines']))
                
                if isinstance(ad['descriptions'], list):
                    top_descriptions.extend(ad['descriptions'])
                else:
                    top_descriptions.append(str(ad['descriptions']))
            
            # Extract creative copy from bottom performers  
            bottom_headlines = []
            bottom_descriptions = []
            
            for idx, ad in bottom_performers.iterrows():
                if isinstance(ad['headlines'], list):
                    bottom_headlines.extend(ad['headlines'])
                else:
                    bottom_headlines.append(str(ad['headlines']))
                
                if isinstance(ad['descriptions'], list):
                    bottom_descriptions.extend(ad['descriptions'])
                else:
                    bottom_descriptions.append(str(ad['descriptions']))
            
            # Extract asset-level creative
            best_asset_copy = []
            learning_asset_copy = []
            
            if not best_assets.empty:
                best_asset_copy = best_assets['asset'].tolist()
            
            if not learning_assets.empty:
                learning_asset_copy = learning_assets['asset'].tolist()
            
            prompt = f"""
            As a Google Ads creative expert, analyze the creative copy patterns between high and low performing elements.
            
            **HIGH-PERFORMING CREATIVE:**
            
            Headlines from high-CTR ads:
            {chr(10).join(f"• {h}" for h in top_headlines[:15])}
            
            Descriptions from high-CTR ads:
            {chr(10).join(f"• {d}" for d in top_descriptions[:10])}
            
            "Best" rated assets:
            {chr(10).join(f"• {a}" for a in best_asset_copy[:10])}
            
            **LOW-PERFORMING CREATIVE:**
            
            Headlines from low-CTR ads:
            {chr(10).join(f"• {h}" for h in bottom_headlines[:15])}
            
            Descriptions from low-CTR ads:
            {chr(10).join(f"• {d}" for d in bottom_descriptions[:10])}
            
            "Learning" rated assets:
            {chr(10).join(f"• {a}" for a in learning_asset_copy[:10])}
            
            **ANALYSIS FOCUS:**
            - Language patterns, words, and phrases that drive clicks
            - Emotional triggers and psychological approaches that work
            - Call-to-action effectiveness and messaging style
            - Value proposition clarity and positioning
            - Creative themes and angles that resonate
            
            Please provide your analysis as a valid JSON response with this format:
            {{
                "winning_elements": "Detailed analysis of successful creative patterns and approaches",
                "weak_elements": "Analysis of underperforming creative patterns to avoid",
                "winning_phrases": ["phrase1", "phrase2", "phrase3", "phrase4", "phrase5"],
                "weak_phrases": ["avoid1", "avoid2", "avoid3", "avoid4", "avoid5"],
                "creative_recommendations": "Specific actionable recommendations for future creative copy"
            }}
            
            Return only the JSON response, no additional text.
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            
            import json
            try:
                return json.loads(response.choices[0].message.content)
            except:
                # Fallback if JSON parsing fails
                content = response.choices[0].message.content
                return {
                    "winning_elements": content[:len(content)//4],
                    "weak_elements": content[len(content)//4:len(content)//2],
                    "winning_phrases": ["Data analysis", "Quick quote", "Best prices", "Save money", "Compare now"],
                    "weak_phrases": ["Generic offer", "Standard service", "Basic coverage", "Regular rates", "Normal terms"],
                    "creative_recommendations": content[len(content)//2:]
                }
            
        except Exception as e:
            return {
                "winning_elements": f"Error analyzing creative patterns: {str(e)}",
                "weak_elements": "Unable to analyze weak creative patterns",
                "winning_phrases": [],
                "weak_phrases": [],
                "creative_recommendations": "Please try again or check your OpenAI API key"
            }

    def generate_best_creative_suggestions(self, creative_analysis, creative_insights, target_audience, additional_context, num_variations=3):
        """Generate best creative suggestions based on data-driven insights."""
        try:
            # Extract successful patterns from analysis
            successful_headlines = []
            successful_descriptions = []
            
            for idx, ad in creative_analysis['top_performers'].iterrows():
                if isinstance(ad['headlines'], list):
                    successful_headlines.extend(ad['headlines'][:3])
                else:
                    successful_headlines.append(str(ad['headlines']))
                
                if isinstance(ad['descriptions'], list):
                    successful_descriptions.extend(ad['descriptions'][:2])
                else:
                    successful_descriptions.append(str(ad['descriptions']))
            
            # Get winning phrases if available
            winning_phrases = []
            creative_recs = ""
            if creative_insights:
                winning_phrases = creative_insights.get('winning_phrases', [])
                creative_recs = creative_insights.get('creative_recommendations', '')
            
            # Get best asset copy
            best_asset_copy = []
            if not creative_analysis['best_assets'].empty:
                best_asset_copy = creative_analysis['best_assets']['asset'].tolist()[:5]
            
            prompt = f"""
            Generate {num_variations} BEST CREATIVE SUGGESTIONS for Google Ads based on proven winning patterns.
            
            **PROVEN SUCCESSFUL CREATIVE:**
            Headlines: {' | '.join(successful_headlines[:10])}
            Descriptions: {' | '.join(successful_descriptions[:8])}
            Best Assets: {' | '.join(best_asset_copy)}
            
            **WINNING PHRASES TO USE:**
            {', '.join(winning_phrases[:10])}
            
            **CREATIVE INSIGHTS:**
            {creative_recs}
            
            **TARGET:** {target_audience}
            **CONTEXT:** {additional_context}
            
            **REQUIREMENTS:**
            - Follow proven patterns from the successful creative above
            - Incorporate winning phrases and elements
            - Each suggestion needs 8-12 headlines (max 30 chars each)  
            - Each suggestion needs 3-4 descriptions (max 90 chars each)
            - Focus on data-driven recommendations, not speculation
            - Prioritize approaches that have already shown success
            
            Please return your response as a valid JSON array:
            [
                {{
                    "strategy": "Brief strategy name (e.g. 'Speed + Price Focus')",
                    "headlines": ["headline1", "headline2", ...],
                    "descriptions": ["desc1", "desc2", ...],
                    "reasoning": "Why this approach will work based on the data"
                }}
            ]
            
            Return only the JSON response, no additional text.
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            
            import json
            try:
                result = json.loads(response.choices[0].message.content)
                # Ensure result is a list of dicts
                if isinstance(result, dict):
                    result = [result]
                elif isinstance(result, str):
                    result = self._parse_creative_fallback_suggestions(result, num_variations)
                return result
            except:
                return self._parse_creative_fallback_suggestions(response.choices[0].message.content, num_variations)
            
        except Exception as e:
            return [{"strategy": f"Error: {str(e)}", "headlines": ["Check API key"], "descriptions": ["Generation failed"], "reasoning": "Please try again"}]

    def generate_themed_creative_variants(self, creative_insights, target_audience, additional_context, num_variations=5):
        """Generate themed creative variants testing different messaging angles."""
        try:
            insights_text = ""
            if creative_insights:
                insights_text = f"""
                Winning Elements: {creative_insights.get('winning_elements', 'N/A')}
                Recommendations: {creative_insights.get('creative_recommendations', 'N/A')}
                """
            
            prompt = f"""
            Generate {num_variations} THEMED CREATIVE VARIANTS for Google Ads testing different messaging angles.
            
            **CREATIVE INSIGHTS:**
            {insights_text}
            
            **TARGET:** {target_audience}
            **CONTEXT:** {additional_context}
            
            **VARIANT THEMES TO EXPLORE:**
            1. Price/Cost Focus - Emphasize savings, value, competitive pricing
            2. Speed/Convenience Focus - Quick service, fast quotes, easy process  
            3. Trust/Security Focus - Reliability, protection, peace of mind
            4. Exclusivity/Premium Focus - Special offers, limited time, premium service
            5. Problem/Solution Focus - Address pain points, provide solutions
            
            **REQUIREMENTS:**
            - Each variant should explore a different messaging angle/theme
            - 8-12 headlines per variant (max 30 chars each)
            - 3-4 descriptions per variant (max 90 chars each)
            - Be creative and test new hypotheses while staying relevant
            - Vary emotional triggers and value propositions
            
            Please return your response as a valid JSON array:
            [
                {{
                    "theme": "Theme name (e.g. 'Price Focus', 'Speed Focus')",
                    "headlines": ["headline1", "headline2", ...],
                    "descriptions": ["desc1", "desc2", ...], 
                    "reasoning": "Experimental hypothesis for this themed approach"
                }}
            ]
            
            Return only the JSON response, no additional text.
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            
            import json
            try:
                result = json.loads(response.choices[0].message.content)
                # Ensure result is a list of dicts
                if isinstance(result, dict):
                    result = [result]
                elif isinstance(result, str):
                    result = self._parse_creative_fallback_variants(result, num_variations)
                return result
            except:
                return self._parse_creative_fallback_variants(response.choices[0].message.content, num_variations)
            
        except Exception as e:
            return [{"theme": f"Error: {str(e)}", "headlines": ["Check API key"], "descriptions": ["Generation failed"], "reasoning": "Please try again"}]

    def _parse_creative_fallback_suggestions(self, content, num_variations):
        """Fallback parser for best suggestions when JSON parsing fails."""
        try:
            suggestions = []
            for i in range(num_variations):
                suggestions.append({
                    "strategy": f"Data-Driven Strategy {i+1}",
                    "headlines": [f"Best Headline {i+1}-1", f"Proven Copy {i+1}-2", f"Winner {i+1}-3"],
                    "descriptions": [f"Best description {i+1}-1", f"Proven copy {i+1}-2"],
                    "reasoning": f"Based on winning patterns - fallback {i+1}"
                })
            return suggestions
        except:
            return [{"strategy": "Parsing Error", "headlines": ["Check response"], "descriptions": ["Check response"], "reasoning": "Fallback failed"}]

    def _parse_creative_fallback_variants(self, content, num_variations):
        """Fallback parser for themed variants when JSON parsing fails."""
        try:
            themes = ["Price Focus", "Speed Focus", "Trust Focus", "Premium Focus", "Solution Focus"]
            variants = []
            for i in range(num_variations):
                theme = themes[i] if i < len(themes) else f"Theme {i+1}"
                variants.append({
                    "theme": theme,
                    "headlines": [f"{theme} Headline {i+1}-1", f"{theme} Copy {i+1}-2", f"{theme} Test {i+1}-3"],
                    "descriptions": [f"{theme} description {i+1}-1", f"{theme} copy {i+1}-2"],
                    "reasoning": f"Testing {theme} approach - fallback {i+1}"
                })
            return variants
        except:
            return [{"theme": "Parsing Error", "headlines": ["Check response"], "descriptions": ["Check response"], "reasoning": "Fallback failed"}]

    def categorize_creative_elements(self, ads_data, asset_data):
        """Use LLM to categorize creative elements for better analysis."""
        try:
            # Extract all headlines and descriptions
            all_headlines = []
            all_descriptions = []
            
            for _, ad in ads_data.iterrows():
                if isinstance(ad['headlines'], list):
                    all_headlines.extend(ad['headlines'])
                elif isinstance(ad['headlines'], str) and ad['headlines']:
                    all_headlines.extend(ad['headlines'].split('|'))
                
                if isinstance(ad['descriptions'], list):
                    all_descriptions.extend(ad['descriptions'])
                elif isinstance(ad['descriptions'], str) and ad['descriptions']:
                    all_descriptions.extend(ad['descriptions'].split('|'))
            
            # Add asset headlines/descriptions
            if not asset_data.empty:
                headlines_assets = asset_data[asset_data['asset_type'] == 'Headline']['asset'].tolist()
                descriptions_assets = asset_data[asset_data['asset_type'] == 'Description']['asset'].tolist()
                all_headlines.extend(headlines_assets)
                all_descriptions.extend(descriptions_assets)
            
            # Remove duplicates and empty strings
            unique_headlines = list(set([h.strip() for h in all_headlines if h and h.strip()]))[:20]
            unique_descriptions = list(set([d.strip() for d in all_descriptions if d and d.strip()]))[:15]
            
            prompt = f"""
            Analyze and categorize these Google Ads creative elements for van insurance.
            
            **HEADLINES:**
            {chr(10).join(f"• {h}" for h in unique_headlines)}
            
            **DESCRIPTIONS:**
            {chr(10).join(f"• {d}" for d in unique_descriptions)}
            
            Categorize each creative element by:
            1. **Value Proposition**: Price/Cost, Speed/Convenience, Trust/Security, Quality/Premium, Comparison/Choice
            2. **Emotional Trigger**: Urgency, Fear/Protection, Savings, Convenience, Authority/Expert
            3. **Call-to-Action Type**: Get Quote, Compare, Save, Apply, Buy, Learn More
            4. **Messaging Style**: Direct/Simple, Descriptive/Detailed, Question/Curiosity, Benefit-focused, Feature-focused
            
            Return as JSON:
            {{
                "headlines_analysis": [
                    {{
                        "text": "headline text",
                        "value_proposition": "category",
                        "emotional_trigger": "category", 
                        "cta_type": "category",
                        "messaging_style": "category"
                    }}
                ],
                "descriptions_analysis": [
                    {{
                        "text": "description text",
                        "value_proposition": "category",
                        "emotional_trigger": "category",
                        "cta_type": "category", 
                        "messaging_style": "category"
                    }}
                ],
                "patterns_summary": {{
                    "most_common_value_prop": "category",
                    "most_common_emotional_trigger": "category",
                    "most_common_cta": "category",
                    "messaging_style_distribution": {{"style": "count"}},
                    "insights": "Key insights about creative patterns"
                }}
            }}
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            
            import json
            try:
                return json.loads(response.choices[0].message.content)
            except:
                return {
                    "headlines_analysis": [],
                    "descriptions_analysis": [],
                    "patterns_summary": {
                        "insights": "Error categorizing creative elements"
                    }
                }
            
        except Exception as e:
            return {
                "headlines_analysis": [],
                "descriptions_analysis": [], 
                "patterns_summary": {
                    "insights": f"Error analyzing creative patterns: {str(e)}"
                }
            }

    def analyze_creative_patterns_enhanced(self, top_performers, bottom_performers, best_assets, learning_assets, ads_data):
        """Enhanced creative pattern analysis using categorization."""
        try:
            # First categorize all creative elements
            creative_categories = self.categorize_creative_elements(ads_data, best_assets)
            
            # Extract creative copy from performers for traditional analysis
            top_headlines = []
            top_descriptions = []
            
            for idx, ad in top_performers.iterrows():
                if isinstance(ad['headlines'], list):
                    top_headlines.extend(ad['headlines'])
                elif isinstance(ad['headlines'], str):
                    top_headlines.extend(ad['headlines'].split('|'))
                
                if isinstance(ad['descriptions'], list):
                    top_descriptions.extend(ad['descriptions'])
                elif isinstance(ad['descriptions'], str):
                    top_descriptions.extend(ad['descriptions'].split('|'))
            
            bottom_headlines = []
            bottom_descriptions = []
            
            for idx, ad in bottom_performers.iterrows():
                if isinstance(ad['headlines'], list):
                    bottom_headlines.extend(ad['headlines'])
                elif isinstance(ad['headlines'], str):
                    bottom_headlines.extend(ad['headlines'].split('|'))
                    
                if isinstance(ad['descriptions'], list):
                    bottom_descriptions.extend(ad['descriptions'])
                elif isinstance(ad['descriptions'], str):
                    bottom_descriptions.extend(ad['descriptions'].split('|'))
            
            best_asset_copy = []
            learning_asset_copy = []
            
            if not best_assets.empty:
                best_asset_copy = best_assets['asset'].tolist()
            
            if not learning_assets.empty:
                learning_asset_copy = learning_assets['asset'].tolist()
            
            prompt = f"""
            Analyze creative performance patterns using both performance data and creative categorization.
            
            **CREATIVE CATEGORIZATION INSIGHTS:**
            {creative_categories.get('patterns_summary', {}).get('insights', 'No categorization available')}
            
            Most common value proposition: {creative_categories.get('patterns_summary', {}).get('most_common_value_prop', 'Unknown')}
            Most common emotional trigger: {creative_categories.get('patterns_summary', {}).get('most_common_emotional_trigger', 'Unknown')}
            Most common CTA type: {creative_categories.get('patterns_summary', {}).get('most_common_cta', 'Unknown')}
            
            **HIGH-PERFORMING CREATIVE:**
            Headlines from high-CTR ads: {' | '.join(top_headlines[:10])}
            Descriptions from high-CTR ads: {' | '.join(top_descriptions[:8])}
            "Best" rated assets: {' | '.join(best_asset_copy[:8])}
            
            **LOW-PERFORMING CREATIVE:**
            Headlines from low-CTR ads: {' | '.join(bottom_headlines[:10])}
            Descriptions from low-CTR ads: {' | '.join(bottom_descriptions[:8])}
            "Learning" rated assets: {' | '.join(learning_asset_copy[:8])}
            
            Please provide enhanced analysis combining performance data with creative categorization and return your response as a valid JSON object:
            
            {{
                "winning_elements": "Analysis of successful creative patterns including categories that work",
                "weak_elements": "Analysis of underperforming patterns including categories to avoid",
                "winning_phrases": ["phrase1", "phrase2", "phrase3", "phrase4", "phrase5"],
                "weak_phrases": ["avoid1", "avoid2", "avoid3", "avoid4", "avoid5"],
                "winning_categories": {{
                    "value_propositions": ["category1", "category2"],
                    "emotional_triggers": ["trigger1", "trigger2"], 
                    "cta_types": ["cta1", "cta2"],
                    "messaging_styles": ["style1", "style2"]
                }},
                "avoid_categories": {{
                    "value_propositions": ["category1", "category2"],
                    "emotional_triggers": ["trigger1", "trigger2"],
                    "cta_types": ["cta1", "cta2"], 
                    "messaging_styles": ["style1", "style2"]
                }},
                "creative_recommendations": "Specific actionable recommendations based on both performance and categorization analysis"
            }}
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            
            import json
            try:
                result = json.loads(response.choices[0].message.content)
                # Add the categorization data to the result
                result['creative_categorization'] = creative_categories
                return result
            except:
                return {
                    "winning_elements": "Error analyzing enhanced patterns",
                    "weak_elements": "Unable to analyze weak patterns",
                    "winning_phrases": [],
                    "weak_phrases": [],
                    "winning_categories": {},
                    "avoid_categories": {},
                    "creative_recommendations": "Analysis failed - check API response",
                    "creative_categorization": creative_categories
                }
            
        except Exception as e:
            return {
                "winning_elements": f"Error in enhanced analysis: {str(e)}",
                "weak_elements": "Unable to analyze patterns",
                "winning_phrases": [],
                "weak_phrases": [],
                "winning_categories": {},
                "avoid_categories": {},
                "creative_recommendations": "Please try again",
                "creative_categorization": {}
            }

    def classify_asset_types(self, asset_data):
        """
        Use LLM to classify each headline/description into creative types.
        
        Args:
            asset_data: DataFrame with asset text and performance data
            
        Returns:
            DataFrame with asset classifications added
        """
        try:
            # Extract unique assets to avoid duplicate API calls
            unique_assets = asset_data[['asset', 'asset_type']].drop_duplicates()
            
            if unique_assets.empty:
                return asset_data
            
            # Prepare assets for classification
            headlines = unique_assets[unique_assets['asset_type'] == 'Headline']['asset'].tolist()
            descriptions = unique_assets[unique_assets['asset_type'] == 'Description']['asset'].tolist()
            
            prompt = f"""
            Classify these Google Ads creative elements into specific types. Focus on van insurance/business insurance context.
            
            **HEADLINES TO CLASSIFY:**
            {chr(10).join(f"{i+1}. {h}" for i, h in enumerate(headlines[:20]))}
            
            **DESCRIPTIONS TO CLASSIFY:**
            {chr(10).join(f"{i+1}. {d}" for i, d in enumerate(descriptions[:15]))}
            
            **CLASSIFICATION CATEGORIES:**
            
            **Value Proposition Types:**
            - Price/Cost (savings, competitive pricing, low cost)
            - Speed/Convenience (fast quotes, quick service, easy process)
            - Trust/Security (reliable, protected, established, safe)
            - Quality/Premium (expert, professional, comprehensive)
            - Comparison/Choice (compare, options, best, top-rated)
            
            **Emotional Trigger Types:**
            - Urgency (limited time, act now, don't wait)
            - Fear/Protection (peace of mind, protect, secure, coverage)
            - Savings (save money, discounts, deals)
            - Convenience (simple, easy, hassle-free)
            - Authority/Expert (trusted, professional, experienced)
            
            **Call-to-Action Types:**
            - Get Quote (get quote, free quote, quote now)
            - Compare (compare prices, compare options)
            - Save (save money, save today)
            - Apply/Buy (apply now, buy online, get covered)
            - Learn More (learn more, find out, discover)
            
            **Messaging Style:**
            - Direct (straightforward, clear, simple language)
            - Descriptive (detailed, explanatory, informative)
            - Question (asking questions, curiosity-driven)
            - Benefit-focused (emphasizes outcomes/benefits)
            - Feature-focused (emphasizes product features)
            
            Return as JSON with this structure:
            {{
                "headlines": [
                    {{
                        "text": "headline text",
                        "value_proposition": "category",
                        "emotional_trigger": "category",
                        "cta_type": "category",
                        "messaging_style": "category"
                    }}
                ],
                "descriptions": [
                    {{
                        "text": "description text", 
                        "value_proposition": "category",
                        "emotional_trigger": "category",
                        "cta_type": "category",
                        "messaging_style": "category"
                    }}
                ],
                "classification_summary": {{
                    "most_effective_value_prop": "category based on performance",
                    "most_effective_emotional_trigger": "category",
                    "most_effective_cta": "category",
                    "messaging_insights": "Key insights about what messaging works"
                }}
            }}
            
            Return only the JSON response, no additional text.
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            
            import json
            try:
                classification_result = json.loads(response.choices[0].message.content)
                
                # Create classification mapping
                classification_map = {}
                
                for headline in classification_result.get('headlines', []):
                    classification_map[headline['text']] = {
                        'value_proposition': headline['value_proposition'],
                        'emotional_trigger': headline['emotional_trigger'],
                        'cta_type': headline['cta_type'],
                        'messaging_style': headline['messaging_style']
                    }
                
                for description in classification_result.get('descriptions', []):
                    classification_map[description['text']] = {
                        'value_proposition': description['value_proposition'],
                        'emotional_trigger': description['emotional_trigger'],
                        'cta_type': description['cta_type'],
                        'messaging_style': description['messaging_style']
                    }
                
                # Add classifications to asset_data
                asset_data_with_types = asset_data.copy()
                
                def add_classification(row):
                    classifications = classification_map.get(row['asset'], {})
                    row['value_proposition'] = classifications.get('value_proposition', 'Unknown')
                    row['emotional_trigger'] = classifications.get('emotional_trigger', 'Unknown')
                    row['cta_type'] = classifications.get('cta_type', 'Unknown')
                    row['messaging_style'] = classifications.get('messaging_style', 'Unknown')
                    return row
                
                asset_data_with_types = asset_data_with_types.apply(add_classification, axis=1)
                
                logger.info(f"Classified {len(unique_assets)} unique assets")
                
                return asset_data_with_types, classification_result.get('classification_summary', {})
                
            except json.JSONDecodeError:
                logger.error("Failed to parse LLM classification response")
                return asset_data, {}
                
        except Exception as e:
            logger.error(f"Error classifying asset types: {e}")
            return asset_data, {}

    def generate_creative_insights(self, asset_data_with_types, performance_categories=['Best', 'Good']):
        """
        Generate insights about what creative types perform best.
        
        Args:
            asset_data_with_types: DataFrame with classified assets
            performance_categories: Which performance categories to analyze
            
        Returns:
            Dictionary with creative insights
        """
        try:
            # Filter for high-performing assets
            high_performers = asset_data_with_types[
                asset_data_with_types['performance_category'].isin(performance_categories)
            ]
            
            # Filter for low-performing assets
            low_performers = asset_data_with_types[
                asset_data_with_types['performance_category'].isin(['Low', 'Learning'])
            ]
            
            if high_performers.empty:
                return {"insights": "No high-performing assets found for analysis"}
            
            # Analyze patterns in high vs low performers
            prompt = f"""
            Analyze creative performance patterns based on asset classifications and performance ratings.
            
            **HIGH-PERFORMING ASSETS (Best/Good):**
            
            Headlines ({len(high_performers[high_performers['asset_type'] == 'Headline'])} assets):
            {chr(10).join(f"• {row['asset']} [VP: {row.get('value_proposition', 'N/A')}, ET: {row.get('emotional_trigger', 'N/A')}, CTA: {row.get('cta_type', 'N/A')}]" for _, row in high_performers[high_performers['asset_type'] == 'Headline'].head(10).iterrows())}
            
            Descriptions ({len(high_performers[high_performers['asset_type'] == 'Description'])} assets):
            {chr(10).join(f"• {row['asset']} [VP: {row.get('value_proposition', 'N/A')}, ET: {row.get('emotional_trigger', 'N/A')}, CTA: {row.get('cta_type', 'N/A')}]" for _, row in high_performers[high_performers['asset_type'] == 'Description'].head(8).iterrows())}
            
            **LOW-PERFORMING ASSETS (Low/Learning):**
            
            Headlines ({len(low_performers[low_performers['asset_type'] == 'Headline'])} assets):
            {chr(10).join(f"• {row['asset']} [VP: {row.get('value_proposition', 'N/A')}, ET: {row.get('emotional_trigger', 'N/A')}, CTA: {row.get('cta_type', 'N/A')}]" for _, row in low_performers[low_performers['asset_type'] == 'Headline'].head(8).iterrows())}
            
            Descriptions ({len(low_performers[low_performers['asset_type'] == 'Description'])} assets):
            {chr(10).join(f"• {row['asset']} [VP: {row.get('value_proposition', 'N/A')}, ET: {row.get('emotional_trigger', 'N/A')}, CTA: {row.get('cta_type', 'N/A')}]" for _, row in low_performers[low_performers['asset_type'] == 'Description'].head(6).iterrows())}
            
            **PERFORMANCE METRICS:**
            High performers - Avg CTR: {high_performers['ctr'].mean():.2%}, Avg Conversions: {high_performers['conversions'].mean():.1f}
            Low performers - Avg CTR: {low_performers['ctr'].mean():.2%}, Avg Conversions: {low_performers['conversions'].mean():.1f}
            
            Provide actionable insights about what creative types and approaches work best.
            
            Return as JSON:
            {{
                "winning_creative_types": {{
                    "value_propositions": ["most effective value prop types"],
                    "emotional_triggers": ["most effective emotional triggers"],
                    "cta_types": ["most effective call-to-action types"],
                    "messaging_styles": ["most effective messaging styles"]
                }},
                "avoid_creative_types": {{
                    "value_propositions": ["value prop types to avoid"],
                    "emotional_triggers": ["emotional triggers to avoid"],
                    "cta_types": ["cta types to avoid"],
                    "messaging_styles": ["messaging styles to avoid"]
                }},
                "key_insights": "Main insights about what makes creative effective in this campaign",
                "creative_recommendations": "Specific recommendations for new creative based on this analysis",
                "top_performing_examples": {{
                    "headlines": ["best headline examples"],
                    "descriptions": ["best description examples"]
                }}
            }}
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            
            import json
            try:
                insights = json.loads(response.choices[0].message.content)
                logger.info("Generated creative insights based on asset classifications")
                return insights
                
            except json.JSONDecodeError:
                logger.error("Failed to parse creative insights response")
                return {"insights": "Error parsing insights response"}
                
        except Exception as e:
            logger.error(f"Error generating creative insights: {e}")
            return {"insights": f"Error generating insights: {str(e)}"}

    def generate_new_creatives_with_insights(self, asset_data, creative_insights, target_audience, additional_context, 
                                           use_categories=['Best', 'Good'], num_headlines=30, num_descriptions=30):
        """
        Generate new creatives based on insights and high-performing assets.
        
        Args:
            asset_data: DataFrame with asset data and classifications
            creative_insights: Insights from creative analysis
            target_audience: Target audience description
            additional_context: Additional context/brief
            use_categories: Performance categories to use as input
            num_headlines: Number of headlines to generate
            num_descriptions: Number of descriptions to generate
            
        Returns:
            Dictionary with generated creative and reasoning
        """
        try:
            # Filter for assets in specified categories
            input_assets = asset_data[asset_data['performance_category'].isin(use_categories)]
            
            if input_assets.empty:
                logger.warning("No assets found in specified categories, using all assets")
                input_assets = asset_data
            
            # Extract successful patterns
            successful_headlines = input_assets[input_assets['asset_type'] == 'Headline']['asset'].tolist()
            successful_descriptions = input_assets[input_assets['asset_type'] == 'Description']['asset'].tolist()
            
            # Get insights from creative analysis
            winning_types = creative_insights.get('winning_creative_types', {})
            recommendations = creative_insights.get('creative_recommendations', '')
            key_insights = creative_insights.get('key_insights', '')
            top_examples = creative_insights.get('top_performing_examples', {})
            
            prompt = f"""
            Generate {num_headlines} new headlines and {num_descriptions} new descriptions for Google Ads based on proven successful patterns.
            
            **TARGET AUDIENCE:** {target_audience}
            **ADDITIONAL CONTEXT:** {additional_context}
            
            **SUCCESSFUL CREATIVE PATTERNS TO FOLLOW:**
            
            Winning Value Propositions: {', '.join(winning_types.get('value_propositions', []))}
            Winning Emotional Triggers: {', '.join(winning_types.get('emotional_triggers', []))}
            Winning CTA Types: {', '.join(winning_types.get('cta_types', []))}
            Winning Messaging Styles: {', '.join(winning_types.get('messaging_styles', []))}
            
            **TOP PERFORMING EXAMPLES:**
            Headlines: {' | '.join(top_examples.get('headlines', [])[:5])}
            Descriptions: {' | '.join(top_examples.get('descriptions', [])[:3])}
            
            **SUCCESSFUL ASSET PATTERNS:**
            Successful Headlines ({len(successful_headlines)} examples):
            {chr(10).join(f"• {h}" for h in successful_headlines[:15])}
            
            Successful Descriptions ({len(successful_descriptions)} examples):
            {chr(10).join(f"• {d}" for d in successful_descriptions[:10])}
            
            **KEY INSIGHTS TO APPLY:**
            {key_insights}
            
            **CREATIVE RECOMMENDATIONS:**
            {recommendations}
            
            **REQUIREMENTS:**
            - Generate exactly {num_headlines} unique headlines (max 30 characters each)
            - Generate exactly {num_descriptions} unique descriptions (max 90 characters each)
            - Follow the winning patterns identified in the analysis
            - Use successful value propositions, emotional triggers, and CTAs
            - Maintain relevance to: {target_audience}
            - Incorporate: {additional_context}
            - Ensure variety while staying true to what works
            
            Return as JSON:
            {{
                "headlines": ["{num_headlines} headlines here"],
                "descriptions": ["{num_descriptions} descriptions here"],
                "strategy_summary": "Brief summary of the creative strategy used",
                "pattern_usage": "How you incorporated the successful patterns",
                "character_counts": {{
                    "headlines_avg": "average headline length",
                    "descriptions_avg": "average description length"
                }}
            }}
            
            Return only the JSON response, no additional text.
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            
            import json
            try:
                generated_creative = json.loads(response.choices[0].message.content)
                
                # Validate we got the right number of items
                headlines = generated_creative.get('headlines', [])
                descriptions = generated_creative.get('descriptions', [])
                
                logger.info(f"Generated {len(headlines)} headlines and {len(descriptions)} descriptions")
                
                return generated_creative
                
            except json.JSONDecodeError:
                logger.error("Failed to parse generated creative response")
                return self._generate_fallback_creative_with_insights(
                    successful_headlines, successful_descriptions, target_audience, 
                    num_headlines, num_descriptions
                )
                
        except Exception as e:
            logger.error(f"Error generating new creatives with insights: {e}")
            return self._generate_fallback_creative_with_insights(
                [], [], target_audience, num_headlines, num_descriptions
            )

    def _generate_fallback_creative_with_insights(self, successful_headlines, successful_descriptions, 
                                                target_audience, num_headlines=30, num_descriptions=30):
        """Fallback creative generation when main method fails."""
        try:
            # Create fallback based on successful patterns
            base_headlines = successful_headlines[:10] if successful_headlines else [
                "Get Van Insurance Quote", "Compare Van Insurance", "Save on Van Cover",
                "Professional Van Insurance", "Quick Van Insurance Quote"
            ]
            
            base_descriptions = successful_descriptions[:5] if successful_descriptions else [
                "Get competitive van insurance quotes online. Fast and easy.",
                "Professional van insurance from trusted providers.",
                "Save money on van insurance. Compare quotes today."
            ]
            
            # Generate variations
            headlines = []
            descriptions = []
            
            # Create headline variations
            for i in range(num_headlines):
                base_idx = i % len(base_headlines)
                base = base_headlines[base_idx]
                variation = f"{base} {i+1}" if i >= len(base_headlines) else base
                headlines.append(variation[:30])  # Ensure character limit
            
            # Create description variations  
            for i in range(num_descriptions):
                base_idx = i % len(base_descriptions)
                base = base_descriptions[base_idx]
                variation = f"{base} Option {i+1}." if i >= len(base_descriptions) else base
                descriptions.append(variation[:90])  # Ensure character limit
            
            return {
                "headlines": headlines,
                "descriptions": descriptions,
                "strategy_summary": "Fallback creative generation based on available patterns",
                "pattern_usage": "Used successful examples as base for variations",
                "character_counts": {
                    "headlines_avg": sum(len(h) for h in headlines) / len(headlines),
                    "descriptions_avg": sum(len(d) for d in descriptions) / len(descriptions)
                }
            }
            
        except Exception as e:
            logger.error(f"Fallback creative generation failed: {e}")
            return {
                "headlines": [f"Van Insurance Quote {i+1}" for i in range(num_headlines)],
                "descriptions": [f"Get van insurance quote {i+1}. Save money today." for i in range(num_descriptions)],
                "strategy_summary": "Basic fallback creative",
                "pattern_usage": "No patterns available",
                "character_counts": {"headlines_avg": 20, "descriptions_avg": 45}
            } 