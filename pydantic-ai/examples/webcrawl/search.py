from ddgs import DDGS

results = DDGS().text("python programming", max_results=5)
#print(results)

count = 1
for item in results:
    print(f"Result {count}: \n {item} \n")
    count += 1
