climate_assessment_prompt_zero_shot = """
You are a climate vulnerability and resilience expert. Implement a comprehensive assessment following the IPCC vulnerability framework and multi-scale resilience analysis.

VULNERABILITY FRAMEWORK:
- **Exposure**: Characterize the type of climate or weather hazard.
    • Sudden-Onset → Rapid shocks such as storms, floods, cyclones, or flash droughts
    • Slow-Onset → Gradual stresses such as sea-level rise, prolonged droughts, or heatwaves  
    • Compound → Multiple interacting hazards (e.g., hurricane + flooding + infrastructure failure)  

- **Sensitivity**: Evaluate how strongly the system is affected by the hazard.  
    • Critical → Highly dependent on vulnerable resources; likely severe disruption  
    • Moderate → Some dependence, but buffers exist; disruption noticeable but not catastrophic  
    • Low → Minimal dependence on hazard-affected resources; relatively insulated  

- **Adaptability**: Determine the system’s capacity to respond and recover.  
    • Robust → Strong governance, infrastructure, technology, and social capital; effective recovery likely  
    • Constrained → Some coping mechanisms exist but are limited, uneven, or short-lived  
    • Fragile → Very limited or no capacity to cope; likely overwhelmed without external aid or systemic transformation  

RESILIENCE FRAMEWORK:
- **Temporal Scale**: Choose the primary focus among [short-term absorptive capacity (emergency responses) | medium-term adaptive capacity (policy/infrastructure adjustments) | long-term transformative capacity (systemic redesign/migration)]  
- **Functional System Scale**: Classify the single most affected system based on evidence. Options: [health, energy, food, water, transportation, information]. Consider redundancy, robustness, recovery time, and interdependence.  
- **Spatial Scale**: Choose the primary level among [local | regional | national]. Highlight capacity differences across scales.  

INSTRUCTIONS:
- Always classify using the provided categories only, citing evidence from the document chunk.  
- Ensure all classifications and selections are supported by evidence.  

INPUT:  
Query: {query}  
Retrieved Document Chunk: {context}  

OUTPUT FORMAT (follow this exact structure):  
Region: [Extract/infer geographic region]  
Exposure: [Sudden-Onset | Slow-Onset | Compound]  
Sensitivity: [Critical | Moderate | Low]  
Adaptability: [Robust | Constrained | Fragile]  
Temporal: [short-term absorptive capacity | medium-term adaptive capacity | long-term transformative capacity]  
Functional: [health | energy | food | water | transportation | information]  
Spatial: [local | regional | national]  

EXAMPLE OUTPUT:
Region: Montreal
Exposure: Slow-Onset
Sensitivity: Moderate
Adaptability: Robust
Temporal: medium-term adaptive capacity
Functional: energy
Spatial: regional

Only output in the exact format above, using the exact categories as instructed. Do not include any additional text.
"""

RAG_Answering_prompt = """
You are an expert in climate vulnerability and resilience. Use the retrieved document chunks to provide a comprehensive answer to the query.

INSTRUCTIONS:
- Consider all retrieved document chunks together before answering.
- Synthesize the information; do not repeat text verbatim unless quoting is necessary for evidence.
- Ensure that your answer is **directly relevant to the query** and grounded in the provided context.
- Avoid adding information not supported by the provided chunks.
- If the retrieved chunks are not relevant to the query, should claim there is no mention in the provided contexts.

INPUT:
Query: {query}
Retrieved Document Chunks:
{context}

OUTPUT:
[Provide a concise, evidence-based answer to the query, integrating information from all chunks.】
"""

# 改一下vulnerability framework
# 放top-3 的 chunks retrieved to Generation TASK
generate_ground_truth_with_evidence_prompt = """You are a climate vulnerability and resilience expert. Create a comprehensive assessment following IPCC vulnerability framework and multi-scale resilience analysis.    
    VULNERABILITY FRAMEWORK:
    - **Exposure**: Characterize the type of climate or weather hazard.
        • Sudden-Onset → Rapid shocks such as storms, floods, cyclones, or flash droughts
        • Slow-Onset → Gradual stresses such as sea-level rise, prolonged droughts, or heatwaves  
        • Compound → Multiple interacting hazards (e.g., hurricane + flooding + infrastructure failure)  

    - **Sensitivity**: Evaluate how strongly the system is affected by the hazard.  
        • Critical → Highly dependent on vulnerable resources; likely severe disruption  
        • Moderate → Some dependence, but buffers exist; disruption noticeable but not catastrophic  
        • Low → Minimal dependence on hazard-affected resources; relatively insulated  

    - **Adaptability**: Determine the system’s capacity to respond and recover.  
        • Robust → Strong governance, infrastructure, technology, and social capital; effective recovery likely  
        • Constrained → Some coping mechanisms exist but are limited, uneven, or short-lived  
        • Fragile → Very limited or no capacity to cope; likely overwhelmed without external aid or systemic transformation 

    RESILIENCE FRAMEWORK:
    - **Temporal Scale**: Choose the primary focus among [short-term absorptive capacity (emergency responses) | medium-term adaptive capacity (policy/infrastructure adjustments) | long-term transformative capacity (systemic redesign/migration)]  
    - **Functional System Scale**: Classify the single most affected system based on evidence. Options: [health, energy, food, water, transportation, information]. Consider redundancy, robustness, recovery time, and interdependence.  
    - **Spatial Scale**: Choose the primary level among [local | community | regional | national]. Highlight capacity differences across scales.  

    INSTRUCTIONS:
    - Always classify using the provided categories only, citing evidence from the document chunk.  
    - Ensure all classifications and selections are supported by evidence.  
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
        "type": [Sudden-Onset | Slow-Onset | Compound],
        "evidence": "Direct quotes/paraphrases supporting climate stress assessment"
        }},
        "sensitivity": {{
        "level": [Critical | Moderate | Low],
        "evidence": "Direct quotes/paraphrases supporting system response assessment"
        }},
        "adaptability": {{
        "type": [Robust | Constrained | Fragile],
        "evidence": "Direct quotes/paraphrases supporting adaptive capacity assessment"
        }}
    }},
    "resilience": {{
        "temporal_scale": {{
        "primary_focus": "[short-term absorptive capacity | medium-term adaptive capacity | long-term transformative capacity]",
        "evidence": "Supporting evidence from chunks"
        }},
        "functional_system": {{
        "primary_focus": ["health | energy | food | water | transportation | information"], 
        "evidence": "Supporting evidence from chunks"
        }},
        "spatial_scale": {{
        "primary_focus": "[local | regional | national]",
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

system_prompt = "You are a climate expert who creates structured vulnerability and resilience assessments following IPCC frameworks."
