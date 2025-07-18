climate_assessment_prompt = """You are a climate vulnerability and resilience expert. Create a comprehensive assessment following IPCC vulnerability framework and multi-scale resilience analysis.

VULNERABILITY FRAMEWORK:
- **Exposure**: Assess the degree of climate stress (long-term climate changes, variability, and extreme event magnitude/frequency)
- **Sensitivity**: Evaluate how the system responds to climate change (precondition for vulnerability - higher sensitivity = greater potential impacts)   
- **Adaptability**: Determine capacity to adjust to climate stimuli based on wealth, technology, education, information, skills, infrastructure, resources, and governance

RESILIENCE FRAMEWORK:
- **Temporal Scale**: Choose primary focus among short-term absorptive capacity (emergency responses), medium-term adaptive capacity (policy/infrastructure adjustments), or long-term transformative capacity (systemic redesign/migration)
- **Functional System Scale**: Select 1-3 key affected systems from health, energy, food, water, transportation, information. Consider redundancy, robustness, recovery time, interdependence
- **Spatial Scale**: Choose primary level among local, community, regional, national. Highlight capacity differences across scales

INSTRUCTIONS:
- Use 1-5 scale (1=very low, 5=very high) with evidence from document chunks
- Quote directly from chunks when possible, clearly indicate paraphrasing
- Ensure all scores and selections are supported by evidence, only output as instructed

INPUT:
Query: {query}
Retrieved Document Chunk: {context}

OUTPUT FORMAT (follow this exact structure):
Region: [Extract/infer geographic region]
Exposure: [1-5 or NA]
Sensitivity: [1-5 or NA]
Adaptability: [1-5 or NA]
Temporal_Scale: [short-term absorptive capacity | medium-term adaptive capacity | long-term transformative capacity | NA]
Functional_System: [health, energy, food, water, transportation, information - list 1-3 separated by commas | NA]
Spatial_Scale: [local | community | regional | national | NA]
Answer: [2-3 sentence concise answer addressing query based on chunks]

Only output in the exact format above, and the choice are exactly as instructed. Do not include any additional text.   
"""


generate_ground_truth_with_evidence_prompt = """You are a climate vulnerability and resilience expert. Create a comprehensive assessment following IPCC vulnerability framework and multi-scale resilience analysis.
    VULNERABILITY FRAMEWORK:
    - **Exposure**: Assess the degree of climate stress (long-term climate changes, variability, and extreme event magnitude/frequency)
    - **Sensitivity**: Evaluate how the system responds to climate change (precondition for vulnerability - higher sensitivity = greater potential impacts)   
    - **Adaptability**: Determine capacity to adjust to climate stimuli based on wealth, technology, education, information, skills, infrastructure, resources, and governance
    NOTE: If no evidence is found, output a score of "NA" and "No evidence found." in evidence.

    RESILIENCE FRAMEWORK:
    - **Temporal Scale**: Choose primary focus among short-term absorptive capacity (emergency responses), medium-term adaptive capacity (policy/infrastructure adjustments), or long-term transformative capacity (systemic redesign/migration)
    - **Functional System Scale**: Select 1-3 key affected systems from health, energy, food, water, transportation, information. Consider redundancy, robustness, recovery time, interdependence
    - **Spatial Scale**: Choose primary level among local, community, regional, national. Highlight capacity differences across scales
    NOTE: If no evidence is found, output a primary_focus of "NA" and "No evidence found." in evidence.

    INSTRUCTIONS:
    - Use 1-5 scale (1=very low, 5=very high) with evidence from document chunks
    - Quote directly from chunks when possible, clearly indicate paraphrasing
    - Ensure all scores and selections are supported by evidence

    INPUT:
    Query: {query}
    Retrieved Document Chunk: {context}

    OUTPUT FORMAT (JSON):
    {{
    "region": "[Extract/infer geographic region]",
    "vulnerability": {{
        "exposure": {{
        "score": [1-5],
        "evidence": "Direct quotes/paraphrases supporting climate stress assessment"
        }},
        "sensitivity": {{
        "score": [1-5], 
        "evidence": "Direct quotes/paraphrases supporting system response assessment"
        }},
        "adaptability": {{
        "score": [1-5],
        "evidence": "Direct quotes/paraphrases supporting adaptive capacity assessment"
        }}
    }},
    "resilience": {{
        "temporal_scale": {{
        "primary_focus": "[short-term absorptive capacity | medium-term adaptive capacity | long-term transformative capacity]",
        "evidence": "Supporting evidence from chunks"
        }},
        "functional_system": {{
        "primary_focus": ["[1-3 from: health, energy, food, water, transportation, information]"], 
        "evidence": "Supporting evidence from chunks"
        }},
        "spatial_scale": {{
        "primary_focus": "[local | community | regional | national]",
        "evidence": "Supporting evidence from chunks"
        }}
    }},
    "question_answer": {{
        "question": {query},
        "answer": "[2-3 sentence concise answer addressing query based on chunks]"
    }}
}}
Only output the JSON response. Do not include any additional text and space in the response."""


validate_ground_truth = """
    You are an expert evaluator tasked with validating and updating vulnerability and resilience assessments based on provided context.

    Query: {query}
    
    Context/Passage: {passage}
    
    Current Generated Answer: {generated_answer}
    
    Please carefully review the current answer and:
    1. Validate if the vulnerability scores (exposure, sensitivity, adaptability) are appropriate based on the context
    2. Validate if the resilience assessments are accurate
    3. Update any scores that seem inappropriate or inaccurate
    4. Please be harsh and critical in your evaluation as a human judger
    
    Return the corrected answer in the EXACT same JSON format as provided, but with updated scores if needed:
    
    OUTPUT FORMAT (JSON):
    {{
        "region": "[Extract/infer geographic region]",
        "vulnerability": {{
            "exposure": {{
                "score": [1-5],
                "evidence": "Direct quotes/paraphrases supporting climate stress assessment"
            }},
            "sensitivity": {{
                "score": [1-5], 
                "evidence": "Direct quotes/paraphrases supporting system response assessment"
            }},
            "adaptability": {{
                "score": [1-5],
                "evidence": "Direct quotes/paraphrases supporting adaptive capacity assessment"
            }}
        }},
        "resilience": {{
            "temporal_scale": {{
                "primary_focus": "[short-term absorptive capacity | medium-term adaptive capacity | long-term transformative capacity]",
                "evidence": "Supporting evidence from chunks"
            }},
            "functional_system": {{
                "primary_focus": ["[1-3 from: health, energy, food, water, transportation, information]"], 
                "evidence": "Supporting evidence from chunks"
            }},
            "spatial_scale": {{
                "primary_focus": "[local | community | regional | national]",
                "evidence": "Supporting evidence from chunks"
            }}
        }},
        "question_answer": {{
            "question": "{query}",
            "answer": "[2-3 sentence concise answer addressing query based on chunks]"
        }}
    }}
    
    Important: Return ONLY the JSON, no additional text or explanations.
    NOTE: Please be harsh and critical in your evaluation as a human judger!
    """