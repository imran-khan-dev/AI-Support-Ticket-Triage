with open('tests/test_agent_memory_integration.py', 'r') as f:
    content = f.read()

# Pattern 1: Tests that create session and store memory before route
# Need to move session creation inside the _route context

# Let me read the first few tests to understand the structure
import re

# Find all test functions and their structure
tests = re.findall(r'def (test_\w+)\(database[^)]*\):', content)
print(f"Found {len(tests)} test functions")
for t in tests:
    print(f"  {t}")