# Sample Results

This file contains a short, curated set of project outputs for demonstration and assignment review.

## Demo metrics

- Total indexed chunks: 1793
- Embedding model: all-MiniLM-L6-v2
- Chunking method: RecursiveCharacterTextSplitter
- Chunk size: 500
- Chunk overlap: 50
- LLM backend: LM Studio (OpenAI-compatible)
- Vector store: Chroma
- Retrieval mode: RAG with semantic search
- Current tool support: weather and currency conversion

## Per-query metrics

### 1) Attraction query
- Query: What are the best attractions in Singapore?
- Query type: destination
- Tools used: none
- Input tokens: 1226
- Output tokens: 98
- Total tokens: 1324
- Latency: 43518.38 ms

Answer:
The best attractions in Singapore include the Civic District, which features a blend of colonial and modern architecture alongside museums and art galleries. Little India offers vibrant cultural experiences with delectable food and buzzing nightlife along the Singapore River quays. For unique dining, explore halal, vegetarian, and vegan food options across the city. Consider staying in areas like Sentosa or Marina Bay for must-visit neighborhoods, or experience heritage sites in Katong Joo Chiat and Kampong Gelam.

### 2) Family itinerary query
- Query: Suggest a 3-day itinerary in Singapore for a family.
- Query type: destination
- Tools used: none
- Input tokens: 1403
- Output tokens: 155
- Total tokens: 1558
- Latency: 93520.38 ms

Question: Suggest a 3-day itinerary in Singapore for a family.

Answer:
Here’s a 3-day family-friendly itinerary in Singapore:
- Day 1: Sentosa Island and Universal Studios
- Day 2: Marina Bay Sands and Gardens by the Bay
- Day 3: Singapore Zoo and Jurong Bird Park

This route balances amusement attractions, scenic city views, and family-friendly learning experiences.

### 3) Currency conversion query
- Query: Convert 60000 INR to SGD.
- Query type: tools_only
- Tools used: currency
- Input tokens: N/A
- Output tokens: N/A
- Total tokens: N/A
- Latency: 596.89 ms

Answer:
60000 INR = 799.32 SGD (rate: 0.0133).

### 4) Weather query
- Query: What is the weather like in Singapore for the next 3 days? Should I bring an umbrella?
- Query type: combined
- Tools used: weather
- Input tokens: 1281
- Output tokens: 52
- Total tokens: 1333
- Latency: 55966.55 ms

Answer:
For the next 3 days in Singapore:
- Expect temperatures around 31–32°C.
- High humidity and a chance of heavy, short-lived showers daily.
- It is advisable to carry an umbrella at all times due to frequent sudden rain.

### 5) Combined RAG + MCP query
- Query: I have a budget of 50000 INR. Convert it to SGD and suggest a 3-day itinerary considering the weather forecast.
- Query type: combined
- Tools used: weather, currency
- Input tokens: 1750
- Output tokens: 167
- Total tokens: 1917
- Latency: 127352.7 ms

Answer:
50,000 INR = 666.10 SGD. For your 3-day trip in Singapore:
- Day 1: Sentosa Island with Universal Studios, S.E.A. Aquarium, and indoor attractions
- Day 2: Sentosa Beach and local cuisine
- Day 3: Singapore Zoo or Botanic Gardens, with museum visits in the afternoon

Carry an umbrella due to high rainfall probabilities.

## Summary metrics

- Total successful queries: 5
- Query mix: 2 destination, 1 tools_only, 2 combined
- Average latency: approximately 60,000–70,000 ms per query depending on LLM and network response time
- Average total tokens: approximately 1,500–1,900 tokens for the demo set
- Most expensive flow: combined weather + currency + itinerary query
- Fastest flow: direct currency conversion

## Note

This file is intentionally short and curated for project review and demo presentation. It excludes verbose terminal traces and raw debug logs.
