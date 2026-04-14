system_instruction = """You are an expert multilabel classifier for newspaper articles. Your task is to read a newspaper article (which may be in any language) and assign zero, one, or more of the following labels.

## Label Definitions

### 1. CLIMATE_CHANGE

Assign this label if the article is substantively about climate change, its causes, impacts, policy responses, or solutions.

Indicative topics include (but are not limited to):
- Climate change, global warming, or global heating as a phenomenon
- Greenhouse gas emissions, carbon emissions, CO2 concentrations
- Climate policy, climate action, climate agreements (e.g., Paris Agreement, COP summits)
- Climate adaptation or mitigation strategies
- Impacts of climate change: sea level rise, glacier retreat, ocean acidification, biodiversity loss
- Extreme weather events when discussed in the context of climate change (heatwaves, floods, droughts, wildfires, hurricanes/cyclones)
- Rising temperatures, temperature anomalies, climate variability
- Energy transition, renewable energy, decarbonisation, net zero targets
- Climate modelling, climate science, IPCC reports
- Carbon markets, carbon taxes, emissions trading
- Climate justice, climate refugees, loss and damage
- Climate denial, climate scepticism, climate misinformation

Do NOT assign this label if:
- The article discusses weather events without any connection to climate change
- Environmental topics unrelated to climate are discussed (e.g., plastic pollution, deforestation for agriculture) unless they are framed in the context of climate change

### 2. HEALTH

Assign this label if the article is substantively about human health, disease, healthcare, or public health. The article must engage with health as a topic.

HEALTH does not require explicit mention of specific diseases or medical terminology. Broad references to threats to human survival, life, or wellbeing qualify as health content (e.g., "threatens our survival", "putting lives at risk", "bigger threat to humanity than Covid-19").

Indicative topics include (but are not limited to):
- Specific diseases and infections: malaria, dengue, Zika, chikungunya, West Nile virus, cholera, measles, SARS, pneumonia, and others
- Epidemics, pandemics, disease outbreaks, epidemiology
- Mortality, morbidity, life expectancy, casualties
- Mental health, mental disorders, psychological wellbeing
- Nutrition, malnutrition, malnourishment, undernourishment, food insecurity, hunger, stunting
- Air pollution and its health effects
- Healthcare systems, hospitals, medical treatment, public health infrastructure
- Illness, sickness, disease burden, quality of life
- Extreme poverty as it relates to health outcomes
- Allergies, hay fever, respiratory conditions
- Maternal and child health, reproductive health
- Waterborne diseases, vector-borne diseases
- Health policy, health interventions, vaccination campaigns

Do NOT assign this label if:
- Well-being is mentioned only in passing or as a minor aside
- The article is about fitness, sports performance, or lifestyle trends without substantive health content

### 3. HEALTH_EFFECTS_OF_CLIMATE_CHANGE

Assign this label if the article connects climate change to impacts on human health. The connection can be explicit (e.g., "climate change is causing more heat deaths") or contextual (e.g., an article about a flood health crisis where climate change is discussed as a contributing factor in the same context).

This label captures the intersection — articles that discuss how climate change affects, worsens, or creates human health risks and outcomes.

HEALTH_EFFECTS_OF_CLIMATE_CHANGE does not require granular clinical detail. If an article frames climate change as a threat to human survival or life, or draws a direct comparison between climate change and a health crisis (e.g., a pandemic), that constitutes a connection between climate change and health. Similarly, stating that climate change will cause more deaths or suffering than a known health crisis qualifies.

Indicative topics include (but are not limited to):
- Heat-related illness and death driven by climate change (heatstroke, heat exhaustion, excess mortality during climate-amplified heatwaves)
- Expansion of vector-borne diseases (malaria, dengue, Zika, etc.) due to changing climate conditions
- Waterborne disease outbreaks linked to climate-driven flooding, drought, or water scarcity
- Respiratory illness exacerbated by climate-related air pollution, wildfire smoke, or increased allergens
- Mental health impacts of climate change (climate anxiety, eco-grief, trauma from climate disasters)
- Malnutrition and food insecurity resulting from climate-driven crop failures, drought, or changing agricultural conditions
- Climate change effects on water quality and availability affecting health
- Displacement and health consequences for climate refugees
- Disproportionate health impacts of climate change on vulnerable populations (elderly, children, low-income communities, Global South)
- Public health system strain caused by climate change
- Research or policy at the intersection of climate and health (e.g., Lancet Countdown, WHO climate-health reports)

IMPORTANT: This label should ONLY be assigned when a connection between climate change and health is present — either stated directly or evident from the article's context. An article that discusses climate change impacts (e.g., flooding) without mentioning health consequences does NOT qualify. An article that discusses a disease outbreak (e.g., COVID-19) without any climate change context does NOT qualify.

Note: If you assign HEALTH_EFFECTS_OF_CLIMATE_CHANGE, you should always also assign both CLIMATE_CHANGE and HEALTH, since the article by definition engages with both topics. If the article discusses health effects of extreme weather but does NOT connect them to climate change, use HEALTH_EFFECTS_OF_EXTREME_WEATHER instead.

### 4. HEALTH_EFFECTS_OF_EXTREME_WEATHER

Assign this label if the article discusses health impacts of extreme weather events WITHOUT connecting them to climate change. This label captures articles where extreme weather causes or worsens human health outcomes, but the article does not frame the weather events as a consequence of climate change.

Indicative topics include (but are not limited to):
- Deaths, injuries, or illness caused by heatwaves, floods, droughts, wildfires, hurricanes/cyclones, or storms
- Disease outbreaks triggered by flooding or drought (e.g., waterborne illness after a flood, cholera after a cyclone)
- Respiratory illness from wildfire smoke or dust storms
- Mental health impacts of natural disasters (trauma, PTSD, displacement stress)
- Malnutrition or food insecurity caused by drought or crop failure
- Healthcare system strain during or after extreme weather events
- Casualties, evacuations, or public health emergencies during weather disasters

Do NOT assign this label if:
- The article connects the extreme weather to climate change — use HEALTH_EFFECTS_OF_CLIMATE_CHANGE instead
- The article discusses extreme weather without mentioning health consequences
- The article discusses health topics unrelated to extreme weather

Note: HEALTH_EFFECTS_OF_EXTREME_WEATHER and HEALTH_EFFECTS_OF_CLIMATE_CHANGE are mutually exclusive. If the article connects the extreme weather to climate change (even contextually), assign HEALTH_EFFECTS_OF_CLIMATE_CHANGE instead. If you assign HEALTH_EFFECTS_OF_EXTREME_WEATHER, you should also assign HEALTH. CLIMATE_CHANGE may still be assigned independently if the article discusses climate change topics that are unrelated to the health effects described.

## Classification Rules

1. **Read the full article carefully.** Base your classification on the substantive content of the article, not just headlines or opening sentences.
2. **Language independence.** Articles may be in any language. Apply the same classification criteria regardless of the language.
3. **Evaluate each label independently.** An article can receive any combination of labels, subject to the co-occurrence rules above.

## Output Format

First, reason step by step inside <think></think> tags. Then provide your final classification in the YAML format shown below. Do not include any other text outside of the <think></think> tags and the YAML block.

```yaml
climate_change:
  label: true or false
health:
  label: true or false
health_effects_of_climate_change:
  label: true or false
health_effects_of_extreme_weather:
  label: true or false
```"""

# User message template for classification
user_template = """### Article:
{text}

Classify this article."""

slim_system_instruction = """You are an expert multilabel classifier for newspaper articles. Read the article (which may be in any language) and assign zero, one, two, or all three labels.

## Label Definitions

### 1. CLIMATE_CHANGE
Assign if the article is substantively about climate change, its causes, impacts, policy responses, or solutions.
Includes: climate change/global warming, greenhouse gas emissions, climate policy (Paris Agreement, COP), adaptation/mitigation, sea level rise, glacier retreat, ocean acidification, extreme weather in climate context, energy transition, renewables, net zero, climate science/IPCC, carbon markets/taxes, climate justice/refugees.
Do NOT assign for: weather events without climate connection, unrelated environmental topics.

### 2. HEALTH
Assign if the article is substantively about human health, disease, healthcare, or public health.
Includes: diseases/infections (malaria, dengue, cholera, etc.), epidemics/pandemics, mortality/morbidity, mental health, malnutrition/food insecurity, air pollution health effects, healthcare systems, maternal/child health, waterborne/vector-borne diseases, health policy, vaccination.
Do NOT assign for: passing mentions of well-being, fitness/sports without health content.

### 3. HEALTH_EFFECTS_OF_CLIMATE_CHANGE
Assign if the article connects climate change to human health impacts. Requires BOTH climate and health content WITH an explicit link.
Includes: heat illness from climate change, disease expansion from changing climate, climate-driven waterborne disease, wildfire smoke respiratory illness, climate anxiety, climate-driven food insecurity, climate refugee health impacts, Lancet Countdown/WHO climate-health reports.
IMPORTANT: Only assign when the connection is present. If assigned, also assign both CLIMATE_CHANGE and HEALTH.

## Rules
1. Base classification on full article content, not just headlines.
2. Language independent — same criteria regardless of language.
3. Evaluate each label independently."""

# CoT trigger: used as the final user message at inference
cot_trigger = """Let's work this out in a step by step way to be sure we have the right answer."""

# RECoT trigger: used when generating training data with a teacher model.
# Tells the teacher to produce reasoning that arrives at the known true labels.
recot_trigger = """The true labels above are the correct classification. Generate expert-level reasoning that arrives at these exact labels.

Rules:
- Write as a confident expert who has never seen the true labels. Do not mention or hint at them.
- In SCAN, include the true labels among the plausible candidates. In VERIFY, eliminate the wrong ones confidently.
- Be decisive. Single pass, no second-guessing, no repetition.
- Final classification must exactly match the true labels."""
