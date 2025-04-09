import boto3
import sys
import os

# Add the secrets directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'secrets'))
import my_secrets

# Initialize DynamoDB client
dynamodb = boto3.resource('dynamodb',
    aws_access_key_id=my_secrets.event_agent_aws_access_key_id,
    aws_secret_access_key=my_secrets.event_agent_aws_secret_access_key,
    region_name=my_secrets.event_agent_aws_region
)

def create_table(table_name, key_schema, attribute_definitions):
    """Create a DynamoDB table if it doesn't exist."""
    try:
        # Check if table exists
        existing_tables = dynamodb.meta.client.list_tables()['TableNames']
        if table_name in existing_tables:
            print(f"Table {table_name} already exists.")
            return
        
        # Create table
        table = dynamodb.create_table(
            TableName=table_name,
            KeySchema=key_schema,
            AttributeDefinitions=attribute_definitions,
            BillingMode='PAY_PER_REQUEST'
        )
        
        # Wait for table to be created
        table.meta.client.get_waiter('table_exists').wait(TableName=table_name)
        print(f"Table {table_name} created successfully.")
        
    except Exception as e:
        print(f"Error creating table {table_name}: {e}")
        sys.exit(1)

def main():
    # Create event_emails table
    event_emails_key_schema = [
        {'AttributeName': 'msg_id', 'KeyType': 'HASH'}  # Partition key
    ]
    event_emails_attributes = [
        {'AttributeName': 'msg_id', 'AttributeType': 'S'}
    ]
    create_table('event_emails', event_emails_key_schema, event_emails_attributes)
    
    # Create events table
    events_key_schema = [
        {'AttributeName': 'event_id', 'KeyType': 'HASH'}  # Partition key
    ]
    events_attributes = [
        {'AttributeName': 'event_id', 'AttributeType': 'S'}
    ]
    create_table('events', events_key_schema, events_attributes)

if __name__ == "__main__":
    main() 