"""
Prompt Templates for Shrimp Interpretation System
Expert system prompts for LLM-based shrimp analysis
"""

# Main interpretation prompt template (vision-enabled)
SHRIMP_INTERPRETATION_PROMPT = """You are an expert shrimp aquaculture veterinarian and farm management consultant. Analyze the provided shrimp image and YOLOv8 detection metrics to generate a professional interpretation.

IMAGE ANALYSIS:
Examine the shrimp in the image for:
- Physical appearance (color, shell condition, body structure)
- Behavior indicators (activity level, clustering)
- Visible disease symptoms (spots, discoloration, deformities)
- Size estimation relative to environment

DETECTION DATA:
- Shrimp Count: {count}
- Average Length: {length} cm (Range: {min_length} - {max_length} cm, Uniformity: {uniformity})
- Average Weight: {weight} g (Estimated Sample Biomass: {biomass} g)
- Mean Detection Confidence: {confidence}
- Water Depth: {depth} m
- Pond ID: {pond_id}
- Cycle No: {cycle_no}

HISTORICAL TREND:
{trend}

Based on your image analysis and the provided data, provide a concise 3-4 sentence interpretation covering:

1. **Growth Assessment**: Evaluate if the shrimp size/weight and size uniformity are appropriate for the growth stage
2. **Health Indicators**: Identify any visible disease risks or health concerns from the image and metrics
3. **Recommendations**: Suggest specific actions for feeding, water quality, or disease prevention

Keep the response professional, actionable, and focused on practical aquaculture management. Avoid generic statements."""


# Metrics-only interpretation prompt (used automatically when Ollama has a text LLM without vision)
METRICS_ONLY_INTERPRETATION_PROMPT = """You are an expert shrimp aquaculture veterinarian and farm management consultant. Analyze the YOLOv8 computer-vision detection metrics from a check-tray sample and generate a professional interpretation.

DETECTION METRICS:
- Shrimp Count (in check tray): {count}
- Average Length: {length} cm (Min: {min_length} cm, Max: {max_length} cm, Size Uniformity: {uniformity})
- Average Weight: {weight} g (Estimated Sample Biomass: {biomass} g)
- AI Detection Confidence: {confidence}
- Water Depth: {depth} m
- Pond ID: {pond_id}
- Cycle No: {cycle_no}

HISTORICAL TREND:
{trend}

Provide a concise 3-4 sentence practical assessment covering:
1. **Growth & Size Uniformity**: Evaluate the length-weight profile and size spread for this growth stage.
2. **Stocking & Risk Indicators**: Flag any density, growth-lag, or uniformity concerns visible in the metrics and trend.
3. **Actionable Recommendations**: Give concrete advice on feed adjustment, water quality monitoring, or sampling.

Keep the response direct, specific to the numbers above, and practical for a shrimp farmer."""


# Alternative shorter prompt for faster generation
SHORT_INTERPRETATION_PROMPT = """You are a shrimp farming expert. Analyze this image and data:

- Count: {count} shrimp
- Avg Length: {length} cm  
- Avg Weight: {weight} g
- Trend: {trend}

Provide a 2-3 sentence interpretation covering:
1. Growth stage and health status
2. Key recommendation for the farmer

Be specific and actionable."""


# Disease-focused prompt
DISEASE_FOCUS_PROMPT = """You are a shrimp disease specialist. Examine the image carefully for disease symptoms.

Data:
- Count: {count}
- Avg Length: {length} cm
- Avg Weight: {weight} g

Look for signs of:
- White Spot Syndrome (white patches on shell)
- Early Mortality Syndrome (pale hepatopancreas)
- Vibrio infections (discoloration, lethargy)
- Shell disease (erosions, black spots)

Provide a 3-sentence assessment:
1. Observed symptoms from image
2. Likely disease risk level (low/medium/high)
3. Immediate action recommended"""


# Growth-focused prompt
GROWTH_FOCUS_PROMPT = """You are a shrimp growth optimization specialist.

Current metrics:
- Count: {count}
- Avg Length: {length} cm
- Avg Weight: {weight} g
- Water Depth: {depth} m

Historical trend: {trend}

Assess:
1. Is growth rate on track? (Compare length/weight ratio)
2. Stocking density appropriateness
3. Feeding strategy recommendation

Provide 3-4 sentences of expert guidance."""


def get_prompt_template(template_type: str = "main") -> str:
    """
    Get prompt template by type
    
    Args:
        template_type: Type of template ("main", "short", "disease", "growth")
        
    Returns:
        Prompt template string
    """
    templates = {
        "main": SHRIMP_INTERPRETATION_PROMPT,
        "metrics": METRICS_ONLY_INTERPRETATION_PROMPT,
        "short": SHORT_INTERPRETATION_PROMPT,
        "disease": DISEASE_FOCUS_PROMPT,
        "growth": GROWTH_FOCUS_PROMPT
    }
    
    return templates.get(template_type, SHRIMP_INTERPRETATION_PROMPT)
