with open('tests/test_agent_memory_integration.py', 'r') as f:
    content = f.read()

# Replace all create_session(database) with tenant_id version
content = content.replace('create_session(database)', 'create_session(database, tenant_id="00000000-0000-0000-0000-000000000000")')

with open('tests/test_agent_memory_integration.py', 'w') as f:
    f.write(content)

print('Done')