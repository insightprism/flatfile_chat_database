# Database Design Patterns for Scalable Applications

## Introduction
This document outlines proven database design patterns for building scalable, maintainable applications. These patterns address common challenges in data modeling, performance, and system architecture.

## Multi-Tenancy Patterns

### Pattern 1: Shared Database, Shared Schema
**Use Case**: Cost-effective solution for many small tenants

**Implementation**:
```sql
CREATE TABLE users (
    id UUID PRIMARY KEY,
    tenant_id UUID NOT NULL,
    email VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(tenant_id, email)
);

-- Row-level security
CREATE POLICY tenant_isolation ON users
    FOR ALL TO application_role
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
```

**Pros**: Cost-effective, easy to manage
**Cons**: Security concerns, noisy neighbor problems

### Pattern 2: Database Per Tenant
**Use Case**: High security requirements, large tenants

**Implementation**:
- Separate database for each tenant
- Connection routing based on tenant identifier
- Automated provisioning and management

**Pros**: Complete isolation, customizable per tenant
**Cons**: Higher operational overhead, resource inefficiency

## Hierarchical Data Patterns

### Adjacency List
**Best For**: Simple hierarchies, frequent updates

```sql
CREATE TABLE categories (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255),
    parent_id INTEGER REFERENCES categories(id)
);
```

### Nested Sets
**Best For**: Read-heavy workloads, complex queries

```sql
CREATE TABLE categories (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255),
    lft INTEGER NOT NULL,
    rgt INTEGER NOT NULL
);

-- Find all descendants
SELECT * FROM categories 
WHERE lft > parent.lft AND rgt < parent.rgt;
```

### Closure Table
**Best For**: Complex hierarchies, flexible queries

```sql
CREATE TABLE category_paths (
    ancestor_id INTEGER NOT NULL,
    descendant_id INTEGER NOT NULL,
    depth INTEGER NOT NULL,
    PRIMARY KEY (ancestor_id, descendant_id)
);

-- All descendants at any depth
SELECT c.* FROM categories c
JOIN category_paths p ON c.id = p.descendant_id
WHERE p.ancestor_id = ?;
```

## Performance Patterns

### Read Replicas
- Separate read and write workloads
- Route analytical queries to replicas
- Handle replication lag appropriately

### Partitioning Strategies

#### Horizontal Partitioning (Sharding)
```sql
-- Range partitioning by date
CREATE TABLE orders_2023 PARTITION OF orders
    FOR VALUES FROM ('2023-01-01') TO ('2024-01-01');

-- Hash partitioning by user_id
CREATE TABLE users_0 PARTITION OF users
    FOR VALUES WITH (MODULUS 4, REMAINDER 0);
```

#### Vertical Partitioning
- Split frequently accessed columns from rarely accessed ones
- Separate hot data from cold data
- Use different storage engines optimized for access patterns

### Caching Patterns

#### Cache-Aside
```python
def get_user(user_id):
    # Try cache first
    user = cache.get(f"user:{user_id}")
    if user is None:
        # Cache miss - fetch from database
        user = db.query("SELECT * FROM users WHERE id = ?", user_id)
        cache.set(f"user:{user_id}", user, ttl=3600)
    return user
```

#### Write-Through
```python
def update_user(user_id, data):
    # Update database first
    db.execute("UPDATE users SET ... WHERE id = ?", user_id, data)
    # Then update cache
    cache.set(f"user:{user_id}", data, ttl=3600)
```

## Data Modeling Patterns

### Event Sourcing
Store all changes as events rather than current state:

```sql
CREATE TABLE events (
    id UUID PRIMARY KEY,
    aggregate_id UUID NOT NULL,
    event_type VARCHAR(100) NOT NULL,
    event_data JSONB NOT NULL,
    version INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Rebuild current state from events
SELECT aggregate_id, 
       array_agg(event_data ORDER BY version) as event_history
FROM events 
WHERE aggregate_id = ?
GROUP BY aggregate_id;
```

### CQRS (Command Query Responsibility Segregation)
- Separate models for reads and writes
- Optimized read models for different query patterns
- Eventual consistency between command and query sides

### Audit Patterns

#### Temporal Tables
```sql
CREATE TABLE users (
    id UUID PRIMARY KEY,
    email VARCHAR(255),
    valid_from TIMESTAMP DEFAULT NOW(),
    valid_to TIMESTAMP DEFAULT 'infinity'
);

-- Query historical data
SELECT * FROM users 
WHERE id = ? AND valid_from <= ? AND valid_to > ?;
```

## Migration Patterns

### Blue-Green Deployments
1. Deploy new schema version alongside old
2. Gradually migrate data to new version
3. Switch application to use new version
4. Remove old version when safe

### Backward Compatible Changes
- Add columns with defaults
- Create new tables before removing old ones
- Use views to maintain API compatibility

### Zero-Downtime Migrations
```sql
-- Step 1: Add new column with default
ALTER TABLE users ADD COLUMN new_field VARCHAR(255) DEFAULT 'default_value';

-- Step 2: Backfill existing data (in batches)
UPDATE users SET new_field = calculate_value(old_field) 
WHERE id BETWEEN ? AND ?;

-- Step 3: Update application to use new column
-- Step 4: Remove old column
ALTER TABLE users DROP COLUMN old_field;
```

## Security Patterns

### Row-Level Security (RLS)
```sql
-- Enable RLS
ALTER TABLE sensitive_data ENABLE ROW LEVEL SECURITY;

-- Create policies
CREATE POLICY user_data_policy ON sensitive_data
    FOR ALL TO app_role
    USING (user_id = current_user_id());
```

### Data Encryption
- Encrypt sensitive data at rest
- Use application-level encryption for highly sensitive fields
- Implement proper key management

## Monitoring and Observability

### Query Performance Tracking
- Log slow queries automatically
- Track query patterns and frequency
- Monitor resource utilization per query type

### Data Quality Monitoring
```sql
-- Check for data quality issues
SELECT 
    COUNT(*) as total_records,
    COUNT(CASE WHEN email IS NULL THEN 1 END) as missing_emails,
    COUNT(CASE WHEN created_at > NOW() THEN 1 END) as future_dates
FROM users;
```

## Best Practices Summary

1. **Design for Your Access Patterns**: Optimize for how data will be queried
2. **Plan for Scale**: Consider partitioning and sharding early
3. **Monitor Everything**: Query performance, data quality, resource usage
4. **Test Migrations**: Always test schema changes on production-like data
5. **Document Decisions**: Keep architectural decision records (ADRs)
6. **Security by Design**: Implement security patterns from the start
7. **Measure Twice, Cut Once**: Profile before optimizing

These patterns provide a foundation for building robust, scalable database architectures. Choose patterns based on your specific requirements, scale, and constraints.
