#!/usr/bin/env python3
"""
Automated Demo Script for Flatfile Chat Database

This script provides a comprehensive, automated demonstration of all
major features of the flatfile chat database system. Perfect for
presentations, testing, or getting a quick overview of capabilities.
"""

import asyncio
import sys
import os
from pathlib import Path
from datetime import datetime
import json
import time
import numpy as np
from typing import List, Dict, Any

# Add parent directory to path
sys.path.append('..')

# Add PrismMind directory to path (if available)
prismmind_path = '/home/markly2/prismmind'
if os.path.exists(prismmind_path):
    sys.path.append(prismmind_path)
    print(f"✅ Added PrismMind path: {prismmind_path}")
else:
    print(f"⚠️ PrismMind not found at: {prismmind_path} - will use legacy document processing")

from storage import StorageManager
from config import StorageConfig
from models import Message, Session, Document, UserProfile, MessageRole
from search import SearchQuery, AdvancedSearchEngine
from vector_storage import FlatfileVectorStorage
from document_pipeline import DocumentRAGPipeline


class AutomatedDemo:
    """Automated demo runner for the flatfile chat database."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.demo_data_path = Path("./demo_data_automated")
        self.config = None
        self.storage_manager = None
        self.search_engine = None
        self.vector_storage = None
        self.doc_pipeline = None
        
        # Demo data
        self.users = []
        self.sessions = []
        self.documents = []
        self.messages = []
        
    def log(self, message: str, level: str = "INFO"):
        """Log a message with timestamp."""
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            emoji = {"INFO": "ℹ️", "SUCCESS": "✅", "ERROR": "❌", "WARN": "⚠️"}.get(level, "📝")
            print(f"[{timestamp}] {emoji} {message}")
            
    def separator(self, title: str = ""):
        """Print a section separator."""
        if self.verbose:
            print("\n" + "="*60)
            if title:
                print(f"  {title}")
                print("="*60)
            print()
            
    async def initialize(self):
        """Initialize the demo environment."""
        self.separator("INITIALIZING FLATFILE CHAT DATABASE DEMO")
        
        self.log("Setting up demo environment...")
        
        # Clean up and create demo directory
        import shutil
        if self.demo_data_path.exists():
            shutil.rmtree(self.demo_data_path)
        self.demo_data_path.mkdir(parents=True)
        
        # Setup configuration
        self.config = StorageConfig()
        self.config.storage_base_path = str(self.demo_data_path)
        self.config.enable_compression = False  # Easier to inspect
        self.config.enable_file_locking = True
        
        # Initialize components
        self.storage_manager = StorageManager(self.config)
        self.search_engine = AdvancedSearchEngine(self.config)
        self.vector_storage = FlatfileVectorStorage(self.config)
        self.doc_pipeline = DocumentRAGPipeline(self.config)
        
        self.log(f"Demo data directory: {self.demo_data_path}")
        self.log(f"Configuration: compression={self.config.enable_compression}, locking={self.config.enable_file_locking}")
        self.log("All components initialized successfully!", "SUCCESS")
        
    async def create_demo_users(self):
        """Create demonstration users."""
        self.separator("CREATING DEMO USERS")
        
        demo_users = [
            {
                "user_id": "alice_researcher",
                "username": "Dr. Alice Johnson",
                "preferences": {"theme": "dark", "language": "en"},
                "metadata": {
                    "role": "AI Researcher",
                    "department": "Machine Learning Lab",
                    "expertise": ["deep_learning", "nlp", "computer_vision"],
                    "institution": "Tech University"
                }
            },
            {
                "user_id": "bob_developer",
                "username": "Bob Smith",
                "preferences": {"theme": "light", "language": "en"},
                "metadata": {
                    "role": "Senior Developer",
                    "department": "Engineering",
                    "expertise": ["python", "databases", "api_design"],
                    "company": "DataTech Corp"
                }
            },
            {
                "user_id": "carol_analyst",
                "username": "Carol Williams",
                "preferences": {"theme": "auto", "language": "en"},
                "metadata": {
                    "role": "Data Analyst",
                    "department": "Analytics",
                    "expertise": ["statistics", "visualization", "sql"],
                    "company": "Analytics Inc"
                }
            }
        ]
        
        for user_data in demo_users:
            await self.storage_manager.create_user(
                user_data["user_id"], 
                {
                    "username": user_data["username"],
                    "preferences": user_data["preferences"],
                    "metadata": user_data["metadata"]
                }
            )
            
            # Create a UserProfile object for tracking
            profile = UserProfile(
                user_id=user_data["user_id"],
                username=user_data["username"],
                preferences=user_data["preferences"],
                metadata=user_data["metadata"]
            )
            self.users.append(profile)
            
            self.log(f"Created user: {profile.username} ({profile.user_id})")
            
        self.log(f"Successfully created {len(self.users)} demo users", "SUCCESS")
        
    async def create_demo_sessions(self):
        """Create demonstration chat sessions."""
        self.separator("CREATING CHAT SESSIONS")
        
        session_configs = [
            {
                "user_id": "alice_researcher",
                "title": "Deep Learning Research Discussion",
                "description": "Technical discussion about neural networks and model architectures"
            },
            {
                "user_id": "bob_developer", 
                "title": "Database Design Consultation",
                "description": "Planning and implementing a scalable database solution"
            },
            {
                "user_id": "carol_analyst",
                "title": "Data Analysis Strategy",
                "description": "Statistical analysis and visualization planning"
            },
            {
                "user_id": "alice_researcher",
                "title": "Paper Review Session",
                "description": "Reviewing recent papers in machine learning"
            }
        ]
        
        for session_config in session_configs:
            session_id = await self.storage_manager.create_session(
                user_id=session_config["user_id"],
                title=session_config["title"]
            )
            
            # Create a session object for tracking
            from models import Session
            session = Session(
                session_id=session_id,
                user_id=session_config["user_id"],
                title=session_config["title"],
                metadata={"description": session_config["description"]}
            )
            self.sessions.append(session)
            
            self.log(f"Created session: '{session.title}' for {session_config['user_id']}")
            
        self.log(f"Successfully created {len(self.sessions)} chat sessions", "SUCCESS")
        
    async def populate_chat_messages(self):
        """Add realistic chat messages to sessions."""
        self.separator("POPULATING CHAT MESSAGES")
        
        # Messages for Deep Learning Research Discussion
        dl_messages = [
            Message(role=MessageRole.USER, content="I'm working on a new transformer architecture for document understanding. Can you help me think through the attention mechanism design?"),
            Message(role=MessageRole.ASSISTANT, content="I'd be happy to help with your transformer architecture! Document understanding often benefits from hierarchical attention patterns. Are you planning to use standard multi-head attention, or are you considering something like sparse attention or local-global patterns?"),
            Message(role=MessageRole.USER, content="I was thinking about combining local attention for fine-grained features with global attention for document-level context. But I'm concerned about computational complexity."),
            Message(role=MessageRole.ASSISTANT, content="That's a smart approach! You could implement a hybrid attention mechanism:\n\n1. **Local Attention**: Process text in sliding windows for detailed linguistic features\n2. **Global Attention**: Use a smaller set of learnable global tokens that attend to the full document\n3. **Cross-Attention**: Let local and global representations interact\n\nFor efficiency, consider:\n- Gradient checkpointing for memory\n- Mixed precision training\n- Attention window sizes that divide evenly into your sequence length\n\nWhat's your target document length?"),
            Message(role=MessageRole.USER, content="Documents are typically 5-50 pages, so sequence lengths around 10k-100k tokens. I'm also wondering about position encodings for such long sequences."),
            Message(role=MessageRole.ASSISTANT, content="For sequences that long, standard sinusoidal position encodings can struggle. Consider these alternatives:\n\n**Relative Position Encodings**: More robust for varying lengths\n- T5-style relative position bias\n- RoPE (Rotary Position Embedding)\n\n**Hierarchical Positions**: Encode page/paragraph/sentence structure\n- Separate embeddings for each hierarchy level\n- Sum or concatenate hierarchical position features\n\n**Learnable Position Encodings**: With proper initialization\n- Initialize with sinusoidal patterns\n- Allow fine-tuning for your specific domain\n\nGiven your document structure, I'd recommend experimenting with hierarchical positions combined with RoPE for the fine-grained attention patterns.")
        ]
        
        # Messages for Database Design Consultation  
        db_messages = [
            Message(role=MessageRole.USER, content="I need to design a database for a multi-tenant SaaS application with complex hierarchical data. What approach would you recommend?"),
            Message(role=MessageRole.ASSISTANT, content="For multi-tenant SaaS with hierarchical data, you have several architecture options:\n\n**Multi-tenancy Strategies:**\n1. **Shared Database, Shared Schema**: Single DB with tenant_id columns\n2. **Shared Database, Separate Schemas**: One DB, schema per tenant\n3. **Separate Databases**: Complete isolation per tenant\n\n**For Hierarchical Data:**\n- Adjacency List (simple parent_id)\n- Nested Sets (for read-heavy workloads)\n- Path Enumeration (materialized paths)\n- Closure Tables (most flexible)\n\nWhat's your expected tenant count and data complexity?"),
            Message(role=MessageRole.USER, content="We're expecting 100-1000 tenants initially, with complex organizational hierarchies (companies -> departments -> teams -> users). Read operations are much more frequent than writes."),
            Message(role=MessageRole.ASSISTANT, content="Perfect! For your use case, I'd recommend:\n\n**Multi-tenancy**: Shared database, shared schema with tenant_id\n- Easier to manage and scale initially\n- Row-level security policies for data isolation\n- Composite indexes on (tenant_id, other_columns)\n\n**Hierarchical Data**: Closure Table approach\n```sql\nCREate TABLE organizational_units (\n  id UUID PRIMARY KEY,\n  tenant_id UUID NOT NULL,\n  name VARCHAR(255),\n  unit_type ENUM('company', 'department', 'team')\n);\n\nCREATE TABLE organizational_paths (\n  tenant_id UUID NOT NULL,\n  ancestor_id UUID NOT NULL,\n  descendant_id UUID NOT NULL,\n  depth INTEGER NOT NULL,\n  PRIMARY KEY (tenant_id, ancestor_id, descendant_id)\n);\n```\n\nThis gives you:\n- O(1) subtree queries\n- Efficient ancestor/descendant lookups\n- Easy to maintain with triggers\n\nWould you like me to elaborate on the indexing strategy?"),
            Message(role=MessageRole.USER, content="Yes, and I'm also curious about handling permissions in this hierarchical structure.")
        ]
        
        # Messages for Data Analysis Strategy
        analysis_messages = [
            Message(role=MessageRole.USER, content="I have a dataset with customer behavior data and need to identify patterns that predict churn. What statistical approaches should I consider?"),
            Message(role=MessageRole.ASSISTANT, content="Customer churn prediction is a classic problem with several effective approaches. Let me suggest a systematic strategy:\n\n**1. Exploratory Data Analysis**\n- Churn rate by customer segments\n- Feature distributions for churned vs retained customers\n- Correlation analysis between features\n- Time series analysis of behavior changes\n\n**2. Feature Engineering**\n- Recency, Frequency, Monetary (RFM) analysis\n- Trend features (declining usage patterns)\n- Interaction terms between key variables\n- Time-based aggregations (last 30/60/90 days)\n\n**3. Statistical Models**\n- Logistic Regression (interpretable baseline)\n- Survival Analysis (time-to-churn)\n- Random Forest (feature importance)\n- XGBoost (usually best performance)\n\nWhat's your dataset size and available features?"),
            Message(role=MessageRole.USER, content="I have about 50,000 customers with 2 years of data. Features include usage metrics, support interactions, billing history, and demographic info."),
            Message(role=MessageRole.ASSISTANT, content="Great dataset size for robust modeling! With 2 years of data, you can implement some powerful techniques:\n\n**Time-based Validation Strategy:**\n```\nTrain: Months 1-18\nValidation: Months 19-21  \nTest: Months 22-24\n```\n\n**Key Feature Engineering:**\n1. **Usage Trends**: Month-over-month changes in key metrics\n2. **Support Interaction Patterns**: Frequency, resolution time, sentiment\n3. **Billing Behavior**: Payment delays, plan changes, discount usage\n4. **Lifecycle Features**: Customer age, onboarding completion\n\n**Model Ensemble Approach:**\n- XGBoost for performance\n- Logistic Regression for interpretability  \n- Survival model for time-to-churn insights\n\n**Evaluation Metrics:**\n- Precision/Recall at different thresholds\n- AUC-ROC and AUC-PR\n- Business metrics: Cost of false positives vs false negatives\n\nWould you like me to elaborate on any specific modeling technique?")
        ]
        
        # Paper Review Session messages
        paper_messages = [
            Message(role=MessageRole.USER, content="I just read the latest Attention is All You Need follow-up papers. The improvements in efficiency are impressive, but I'm wondering about the trade-offs."),
            Message(role=MessageRole.ASSISTANT, content="The recent transformer efficiency improvements are indeed fascinating! Which specific papers are you referring to? Are you looking at:\n\n- **Sparse Attention** patterns (BigBird, Longformer)\n- **Linear Attention** approximations (Performer, Linear Transformer)\n- **Low-rank Approximations** (Linformer, FNet)\n- **Retrieval-based** approaches (RAG, FiD)\n\nEach has different trade-offs in terms of accuracy, memory, and computational requirements."),
            Message(role=MessageRole.USER, content="Primarily the sparse attention work - BigBird and Longformer. I'm curious about when the performance trade-offs become significant."),
            Message(role=MessageRole.ASSISTANT, content="Great choice! Both BigBird and Longformer address the O(n²) attention bottleneck cleverly:\n\n**BigBird's Approach:**\n- Random + Window + Global attention\n- Maintains theoretical expressiveness\n- ~8x memory reduction for long sequences\n\n**Longformer's Approach:**\n- Sliding window + task-specific global attention\n- More flexible global attention pattern\n- Better for document-level tasks\n\n**Performance Trade-offs:**\n- **Minimal impact** on tasks where local context dominates (NER, POS tagging)\n- **Some degradation** on tasks requiring long-range dependencies (some QA, summarization)\n- **Sequence length threshold**: Benefits most apparent >4k tokens\n\n**Key Insights from Papers:**\n- Window size matters more than random connections\n- Global attention placement is task-dependent\n- Pre-training with sparse patterns crucial for best results\n\nWhat specific tasks are you considering these for?")
        ]
        
        # Store messages for each session
        message_sets = [
            (self.sessions[0], dl_messages),      # Alice's Deep Learning session
            (self.sessions[1], db_messages),      # Bob's Database session  
            (self.sessions[2], analysis_messages), # Carol's Analysis session
            (self.sessions[3], paper_messages)    # Alice's Paper Review session
        ]
        
        total_messages = 0
        for session, messages in message_sets:
            self.log(f"Adding {len(messages)} messages to '{session.title}'")
            
            for message in messages:
                await self.storage_manager.add_message(
                    session.user_id,
                    session.session_id, 
                    message
                )
                self.messages.append((session, message))
                total_messages += 1
                
        self.log(f"Successfully added {total_messages} messages across all sessions", "SUCCESS")
        
    async def create_demo_documents(self):
        """Create and store demonstration documents."""
        self.separator("CREATING DEMO DOCUMENTS")
        
        # Technical documents with realistic content
        documents_data = [
            {
                "filename": "transformer_architecture_guide.md",
                "content": """# Transformer Architecture Guide

## Introduction
Transformers have revolutionized natural language processing through their attention mechanism. This guide covers key architectural components and design decisions.

## Core Components

### Multi-Head Attention
The attention mechanism allows models to focus on different parts of the input sequence:

```python
def multi_head_attention(Q, K, V, d_model, num_heads):
    d_k = d_model // num_heads
    # Split into multiple heads
    Q_heads = split_heads(Q, num_heads)
    K_heads = split_heads(K, num_heads) 
    V_heads = split_heads(V, num_heads)
    
    # Apply attention to each head
    attention_outputs = []
    for i in range(num_heads):
        attention_output = scaled_dot_product_attention(
            Q_heads[i], K_heads[i], V_heads[i]
        )
        attention_outputs.append(attention_output)
    
    # Concatenate and project
    concat_output = concatenate(attention_outputs)
    return linear_projection(concat_output)
```

### Position Encodings
Since transformers have no inherent notion of sequence order, position encodings are crucial:

- **Sinusoidal Encodings**: Original approach, works well for most tasks
- **Learned Encodings**: Can be more task-specific but less generalizable
- **Relative Position**: T5-style, better for varying sequence lengths

### Feed-Forward Networks
Each transformer layer includes a position-wise feed-forward network:
- Two linear transformations with ReLU activation
- Dimension is typically 4x the model dimension
- Applied identically to each position

## Architectural Variants

### Encoder-Only (BERT-style)
- Bidirectional context
- Best for understanding tasks
- Uses masked language modeling for pre-training

### Decoder-Only (GPT-style)  
- Autoregressive generation
- Best for generation tasks
- Uses causal masking during training

### Encoder-Decoder (T5-style)
- Combines both approaches
- Flexible for many task types
- Can handle variable input/output lengths

## Design Considerations

### Scaling Laws
Research shows predictable relationships between:
- Model size (parameters)
- Dataset size (tokens)
- Compute budget (FLOPs)
- Final performance

### Efficiency Improvements
- **Sparse Attention**: Reduce O(n²) complexity
- **Mixed Precision**: Use FP16 for training
- **Gradient Checkpointing**: Trade compute for memory
- **Model Parallelism**: Scale beyond single GPU limits

## Best Practices

1. **Layer Normalization**: Pre-norm generally works better than post-norm
2. **Dropout**: Apply to attention weights and feed-forward layers
3. **Learning Rate Scheduling**: Warmup followed by decay
4. **Weight Initialization**: Careful initialization crucial for deep models
5. **Regularization**: Label smoothing, weight decay, dropout

## Recent Advances

### Efficient Attention Mechanisms
- **Linformer**: Linear complexity through low-rank approximation
- **Performer**: Uses random features for linear attention
- **BigBird**: Sparse attention with global, local, and random connections

### Architecture Improvements
- **Switch Transformer**: Sparse expert layers
- **PaLM**: Scaling to 540B parameters
- **GLaM**: Mixture of experts with 1.2T parameters

## Conclusion
Transformer architectures continue to evolve rapidly. Key trends include:
- Scaling to larger sizes
- Improving efficiency
- Adding multimodal capabilities
- Better few-shot learning abilities

The field moves quickly, but understanding these fundamentals provides a solid foundation for working with any transformer variant.
""",
                "metadata": {"topic": "deep_learning", "type": "technical_guide", "author": "alice_researcher"}
            },
            
            {
                "filename": "database_design_patterns.md", 
                "content": """# Database Design Patterns for Scalable Applications

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
""",
                "metadata": {"topic": "database_design", "type": "technical_guide", "author": "bob_developer"}
            },
            
            {
                "filename": "statistical_analysis_methods.md",
                "content": """# Statistical Analysis Methods for Data Science

## Introduction
This comprehensive guide covers statistical methods commonly used in data science, from basic descriptive statistics to advanced modeling techniques.

## Descriptive Statistics

### Central Tendency
- **Mean**: Best for symmetric distributions
- **Median**: Robust to outliers, good for skewed data
- **Mode**: Useful for categorical data

### Variability Measures
- **Standard Deviation**: Shows spread in same units as data
- **Variance**: Useful for mathematical operations
- **Interquartile Range (IQR)**: Robust measure of spread
- **Coefficient of Variation**: Allows comparison across different scales

### Distribution Shape
- **Skewness**: Measures asymmetry
- **Kurtosis**: Measures tail heaviness
- **Normality Tests**: Shapiro-Wilk, Kolmogorov-Smirnov

## Inferential Statistics

### Hypothesis Testing Framework
1. **State Hypotheses**: H₀ (null) and H₁ (alternative)
2. **Choose Significance Level**: Typically α = 0.05
3. **Select Test Statistic**: Based on data type and assumptions
4. **Calculate p-value**: Probability of observing data given H₀ is true
5. **Make Decision**: Reject H₀ if p-value < α

### Common Statistical Tests

#### Parametric Tests
```python
# t-test for comparing means
from scipy import stats

# One-sample t-test
t_stat, p_value = stats.ttest_1samp(sample_data, population_mean)

# Two-sample t-test (independent)
t_stat, p_value = stats.ttest_ind(group1, group2)

# Paired t-test (dependent)
t_stat, p_value = stats.ttest_rel(before, after)

# ANOVA for comparing multiple groups
f_stat, p_value = stats.f_oneway(group1, group2, group3)
```

#### Non-Parametric Tests
```python
# Mann-Whitney U test (non-parametric alternative to t-test)
u_stat, p_value = stats.mannwhitneyu(group1, group2)

# Wilcoxon signed-rank test (paired alternative)
w_stat, p_value = stats.wilcoxon(before, after)

# Kruskal-Wallis test (alternative to ANOVA)
h_stat, p_value = stats.kruskal(group1, group2, group3)
```

## Regression Analysis

### Linear Regression
**Assumptions**:
1. Linearity
2. Independence
3. Homoscedasticity (constant variance)
4. Normality of residuals

**Model Diagnostics**:
```python
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Fit model
X = sm.add_constant(X)  # Add intercept
model = sm.OLS(y, X).fit()

# Diagnostic plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Residuals vs fitted
axes[0,0].scatter(model.fittedvalues, model.resid)
axes[0,0].set_title('Residuals vs Fitted')

# Q-Q plot for normality
sm.qqplot(model.resid, line='s', ax=axes[0,1])

# Scale-Location plot
axes[1,0].scatter(model.fittedvalues, np.sqrt(np.abs(model.resid)))
axes[1,0].set_title('Scale-Location')

# Residuals vs leverage
axes[1,1].scatter(model.get_influence().hat_matrix_diag, model.resid)
axes[1,1].set_title('Residuals vs Leverage')
```

### Logistic Regression
For binary outcomes:
```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

# Fit model
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# Predictions
y_pred = logreg.predict(X_test)
y_pred_proba = logreg.predict_proba(X_test)[:, 1]

# Evaluation
print(classification_report(y_test, y_pred))
print(f"AUC-ROC: {roc_auc_score(y_test, y_pred_proba):.3f}")
```

## Time Series Analysis

### Components of Time Series
1. **Trend**: Long-term movement
2. **Seasonality**: Regular patterns
3. **Cyclicality**: Irregular long-term patterns
4. **Noise**: Random variation

### Decomposition
```python
from statsmodels.tsa.seasonal import seasonal_decompose

# Decompose time series
decomposition = seasonal_decompose(ts_data, model='additive')
decomposition.plot()
```

### Stationarity Testing
```python
from statsmodels.tsa.stattools import adfuller

# Augmented Dickey-Fuller test
result = adfuller(ts_data)
print(f'ADF Statistic: {result[0]:.4f}')
print(f'p-value: {result[1]:.4f}')
```

### ARIMA Modeling
```python
from statsmodels.tsa.arima.model import ARIMA

# Fit ARIMA model
model = ARIMA(ts_data, order=(1,1,1))
fitted_model = model.fit()

# Forecasting
forecast = fitted_model.forecast(steps=12)
```

## Survival Analysis

### Kaplan-Meier Estimation
```python
from lifelines import KaplanMeierFitter

kmf = KaplanMeierFitter()
kmf.fit(durations, event_observed)

# Plot survival curve
kmf.plot_survival_function()
```

### Cox Proportional Hazards Model
```python
from lifelines import CoxPHFitter

cph = CoxPHFitter()
cph.fit(df, duration_col='T', event_col='E')
cph.print_summary()
```

## Multivariate Analysis

### Principal Component Analysis (PCA)
```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=0.95)  # Retain 95% of variance
X_pca = pca.fit_transform(X_scaled)

# Explained variance
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
```

### Cluster Analysis
```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Evaluate clustering
silhouette_avg = silhouette_score(X_scaled, clusters)
print(f"Silhouette Score: {silhouette_avg:.3f}")
```

## Experimental Design

### A/B Testing
```python
# Power analysis for sample size calculation
from statsmodels.stats.power import ttest_power

# Calculate required sample size
power = ttest_power(effect_size=0.2, nobs=None, alpha=0.05, power=0.8)
print(f"Required sample size per group: {power:.0f}")

# Statistical test for A/B test results
control_conversion = 0.10
treatment_conversion = 0.12
control_n = 1000
treatment_n = 1000

# Chi-square test for proportions
from scipy.stats import chi2_contingency

contingency_table = np.array([
    [control_conversion * control_n, (1 - control_conversion) * control_n],
    [treatment_conversion * treatment_n, (1 - treatment_conversion) * treatment_n]
])

chi2, p_value, dof, expected = chi2_contingency(contingency_table)
```

### Factorial Design
For studying multiple factors simultaneously:
- **2^k Design**: k factors, each at 2 levels
- **Latin Square**: Controls for two sources of variation
- **Split-Plot Design**: Different randomization for different factors

## Bayesian Analysis

### Bayesian Framework
P(H|D) = P(D|H) × P(H) / P(D)

Where:
- P(H|D): Posterior probability
- P(D|H): Likelihood
- P(H): Prior probability
- P(D): Evidence

### Example: Bayesian A/B Testing
```python
import pymc3 as pm

with pm.Model() as model:
    # Priors
    p_A = pm.Beta('p_A', alpha=1, beta=1)
    p_B = pm.Beta('p_B', alpha=1, beta=1)
    
    # Likelihood
    obs_A = pm.Binomial('obs_A', n=n_A, p=p_A, observed=successes_A)
    obs_B = pm.Binomial('obs_B', n=n_B, p=p_B, observed=successes_B)
    
    # Derived quantity
    lift = pm.Deterministic('lift', (p_B - p_A) / p_A)
    
    # Sample
    trace = pm.sample(2000, tune=1000)
```

## Model Validation and Selection

### Cross-Validation
```python
from sklearn.model_selection import cross_val_score, StratifiedKFold

# k-fold cross-validation
cv_scores = cross_val_score(model, X, y, cv=5)
print(f"CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

# Stratified k-fold for imbalanced data
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=skf)
```

### Information Criteria
- **AIC (Akaike Information Criterion)**: Penalizes model complexity
- **BIC (Bayesian Information Criterion)**: Stronger penalty for complexity
- **AICc**: Corrected AIC for small samples

```python
# Calculate AIC for linear regression
n = len(y)
mse = np.mean((y - y_pred)**2)
k = X.shape[1]  # number of parameters
aic = n * np.log(mse) + 2 * k
```

## Statistical Assumptions and Violations

### Common Violations and Solutions

#### Multicollinearity
- **Detection**: Variance Inflation Factor (VIF)
- **Solutions**: Remove correlated predictors, ridge regression, PCA

#### Heteroscedasticity
- **Detection**: Breusch-Pagan test, White test
- **Solutions**: Weighted least squares, robust standard errors

#### Non-normality
- **Detection**: Shapiro-Wilk test, Q-Q plots
- **Solutions**: Transformations, non-parametric tests

#### Autocorrelation
- **Detection**: Durbin-Watson test, ACF plots
- **Solutions**: Add lagged variables, time series models

## Best Practices

1. **Always Visualize First**: Plots reveal patterns statistics might miss
2. **Check Assumptions**: Don't blindly apply statistical tests
3. **Use Appropriate Sample Sizes**: Power analysis is crucial
4. **Multiple Comparisons**: Adjust p-values when testing multiple hypotheses
5. **Effect Size Matters**: Statistical significance ≠ practical significance
6. **Report Confidence Intervals**: More informative than just p-values
7. **Reproducible Research**: Set random seeds, document methodology
8. **Domain Knowledge**: Statistics should complement, not replace, subject expertise

## Advanced Topics

### Machine Learning vs. Traditional Statistics
- **ML Focus**: Prediction accuracy
- **Statistics Focus**: Understanding relationships and uncertainty
- **Hybrid Approaches**: Statistical learning, interpretable ML

### Causal Inference
- **Randomized Experiments**: Gold standard for causality
- **Observational Studies**: Requires careful design
- **Methods**: Instrumental variables, regression discontinuity, propensity score matching

### Big Data Considerations
- **Sampling**: When full data analysis isn't feasible
- **Multiple Testing**: More hypotheses = higher false discovery rate
- **Computational Statistics**: Bootstrap, permutation tests, MCMC

This guide provides a foundation for statistical analysis in data science. Remember that statistical methods are tools - the key is choosing the right tool for your specific problem and data characteristics.
""",
                "metadata": {"topic": "statistics", "type": "reference_guide", "author": "carol_analyst"}
            }
        ]
        
        # Store documents in different sessions
        session_doc_mapping = [
            (self.sessions[0], documents_data[0]),  # Transformer guide in Alice's session
            (self.sessions[1], documents_data[1]),  # Database patterns in Bob's session
            (self.sessions[2], documents_data[2])   # Statistics guide in Carol's session
        ]
        
        for session, doc_data in session_doc_mapping:
            # Use save_document method which handles the content directly
            doc_id = await self.storage_manager.save_document(
                user_id=session.user_id,
                session_id=session.session_id,
                filename=doc_data["filename"],
                content=doc_data["content"],
                metadata=doc_data["metadata"]
            )
            
            # Create a mock document object for tracking with content
            document = type('Document', (), {
                'filename': doc_data["filename"],
                'metadata': doc_data["metadata"],
                'content': doc_data["content"]  # Keep content for vector processing
            })()
            
            self.documents.append((session, document, doc_id))
            self.log(f"Created document: {document.filename} ({len(doc_data['content'])} chars)")
            
        self.log(f"Successfully created {len(self.documents)} technical documents", "SUCCESS")
        
    async def create_vector_embeddings(self):
        """Create vector embeddings for documents (using mock embeddings)."""
        self.separator("CREATING VECTOR EMBEDDINGS")
        
        def create_mock_embedding(text: str, dim: int = 384) -> list:
            """Create reproducible mock embedding based on text hash."""
            hash_val = hash(text) % (2**31)
            np.random.seed(hash_val)
            return np.random.normal(0, 1, dim).tolist()
            
        total_vectors = 0
        
        for session, document, doc_id in self.documents:
            self.log(f"Processing embeddings for: {document.filename}")
            
            # Simple chunking by paragraphs
            paragraphs = [p.strip() for p in document.content.split('\n\n') if p.strip()]
            
            # Filter out very short paragraphs
            chunks = [p for p in paragraphs if len(p) > 50]
            
            # Create embeddings
            embeddings = [create_mock_embedding(chunk) for chunk in chunks]
            
            # Store vectors
            success = await self.vector_storage.store_vectors(
                session_id=session.session_id,
                document_id=doc_id,
                chunks=chunks,
                vectors=embeddings,
                metadata={
                    "document_name": document.filename,
                    "author": document.metadata.get("author"),
                    "topic": document.metadata.get("topic")
                }
            )
            
            if success:
                total_vectors += len(embeddings)
                self.log(f"  → Created {len(embeddings)} embeddings")
            else:
                self.log(f"  → Failed to store embeddings", "ERROR")
                
        self.log(f"Successfully created {total_vectors} vector embeddings", "SUCCESS")
        
    async def demonstrate_search_capabilities(self):
        """Demonstrate both text and vector search capabilities."""
        self.separator("DEMONSTRATING SEARCH CAPABILITIES")
        
        # Text-based searches
        text_searches = [
            {
                "query": "transformer architecture",
                "description": "Technical architecture search"
            },
            {
                "query": "database design patterns",
                "description": "Design patterns search"
            },
            {
                "query": "statistical analysis methods",
                "description": "Statistical methods search"
            },
            {
                "query": "machine learning",
                "description": "Cross-topic ML search"
            }
        ]
        
        self.log("Performing text-based searches...")
        
        for search_config in text_searches:
            self.log(f"  🔍 Searching for: '{search_config['query']}'")
            
            search_query = SearchQuery(
                query=search_config["query"],
                user_id="alice_researcher",  # Search as Alice
                include_documents=True
            )
            
            try:
                results = await self.search_engine.search(search_query)
                self.log(f"    → Found {len(results)} results")
                
                for i, result in enumerate(results[:2], 1):  # Show top 2
                    content_preview = str(result)[:80] + "..." if len(str(result)) > 80 else str(result)
                    self.log(f"      {i}. Result: {content_preview}")
                    
            except Exception as e:
                self.log(f"    → Search failed: {e}", "ERROR")
                
        # Vector-based searches
        self.log("\nPerforming vector similarity searches...")
        
        vector_searches = [
            "How do attention mechanisms work in transformers?",
            "What are the best practices for database scaling?", 
            "Which statistical test should I use for comparing groups?",
            "Explain neural network architectures"
        ]
        
        for search_text in vector_searches:
            self.log(f"  🔢 Vector search: '{search_text}'")
            
            # Create mock query embedding
            query_embedding = create_mock_embedding(search_text)
            
            try:
                # Search in each session that has documents
                for session, _, _ in self.documents:
                    results = await self.vector_storage.search_similar(
                        session_id=session.session_id,
                        query_vector=query_embedding,
                        top_k=3
                    )
                    
                    if results:
                        self.log(f"    → Session '{session.title}': {len(results)} similar chunks")
                        for i, result in enumerate(results[:1], 1):  # Show top result
                            text_preview = result.text[:60] + "..." if len(result.text) > 60 else result.text
                            self.log(f"      {i}. Similarity: {result.similarity:.3f}")
                            self.log(f"         {text_preview}")
                            
            except Exception as e:
                self.log(f"    → Vector search failed: {e}", "ERROR")
                
        self.log("Search demonstrations completed", "SUCCESS")
        
    def create_mock_embedding(self, text: str, dim: int = 384) -> list:
        """Create reproducible mock embedding based on text hash."""
        hash_val = hash(text) % (2**31)
        np.random.seed(hash_val)
        return np.random.normal(0, 1, dim).tolist()
        
    async def show_system_statistics(self):
        """Display comprehensive system statistics."""
        self.separator("SYSTEM STATISTICS AND ANALYSIS")
        
        # User statistics
        self.log(f"👥 Users: {len(self.users)}")
        for user in self.users:
            self.log(f"  • {user.username} ({user.user_id})")
            if user.metadata:
                expertise = user.metadata.get('expertise', [])
                if expertise:
                    self.log(f"    Expertise: {', '.join(expertise)}")
                    
        # Session statistics
        self.log(f"\n💬 Sessions: {len(self.sessions)}")
        for session in self.sessions:
            self.log(f"  • {session.title}")
            self.log(f"    User: {session.user_id} | ID: {session.session_id}")
            
            # Get message count for this session
            session_messages = [m for s, m in self.messages if s.session_id == session.session_id]
            self.log(f"    Messages: {len(session_messages)}")
            
        # Document statistics  
        self.log(f"\n📄 Documents: {len(self.documents)}")
        total_content_length = 0
        for session, document, doc_id in self.documents:
            content_length = len(document.content)
            total_content_length += content_length
            self.log(f"  • {document.filename}")
            self.log(f"    Size: {content_length:,} characters | Topic: {document.metadata.get('topic', 'N/A')}")
            
        # Storage statistics
        total_size = self.get_directory_size(self.demo_data_path)
        file_count = sum([len(files) for r, d, files in os.walk(self.demo_data_path)])
        
        self.log(f"\n💾 Storage Statistics:")
        self.log(f"  Total Size: {total_size:,} bytes ({total_size / 1024:.1f} KB)")
        self.log(f"  Total Files: {file_count}")
        self.log(f"  Average File Size: {total_size / max(file_count, 1):.1f} bytes")
        self.log(f"  Content Size: {total_content_length:,} characters")
        
        # Configuration info
        self.log(f"\n⚙️ Configuration:")
        self.log(f"  Storage Path: {self.config.storage_base_path}")
        self.log(f"  Compression: {self.config.enable_compression}")
        self.log(f"  File Locking: {self.config.enable_file_locking}")
        self.log(f"  Max Message Size: {self.config.max_message_size_bytes:,} bytes")
        
        # Show file structure sample
        self.log(f"\n📁 File Structure Sample:")
        self.print_directory_tree(self.demo_data_path, max_depth=2)
        
    def get_directory_size(self, path):
        """Calculate total size of directory recursively."""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                if os.path.exists(file_path):
                    total_size += os.path.getsize(file_path)
        return total_size
        
    def print_directory_tree(self, path, prefix="", max_depth=3, current_depth=0):
        """Print directory tree structure."""
        if current_depth > max_depth:
            return
            
        path = Path(path)
        if not path.exists():
            return
            
        items = list(path.iterdir())
        items.sort(key=lambda x: (x.is_file(), x.name))
        
        for i, item in enumerate(items[:8]):  # Limit to first 8 items for brevity
            is_last = i == len(items) - 1
            current_prefix = "└── " if is_last else "├── "
            
            if item.is_file():
                size = item.stat().st_size
                self.log(f"{prefix}{current_prefix}{item.name} ({size} bytes)")
            else:
                self.log(f"{prefix}{current_prefix}{item.name}/")
                
                if current_depth < max_depth:
                    next_prefix = prefix + ("    " if is_last else "│   ")
                    self.print_directory_tree(item, next_prefix, max_depth, current_depth + 1)
                    
        if len(items) > 8:
            self.log(f"{prefix}└── ... ({len(items) - 8} more items)")
            
    async def demonstrate_configuration_systems(self):
        """Demonstrate both legacy and new configuration systems."""
        self.separator("CONFIGURATION SYSTEMS DEMONSTRATION")
        
        # Legacy configuration
        self.log("Legacy Configuration System:")
        self.log(f"  Base Path: {self.config.storage_base_path}")
        self.log(f"  Compression: {self.config.enable_compression}")
        self.log(f"  File Locking: {self.config.enable_file_locking}")
        self.log(f"  Max Message Size: {self.config.max_message_size_bytes:,} bytes")
        self.log(f"  Vector Search Top K: {self.config.vector_search_top_k}")
        
        # Try new configuration system
        try:
            from config_new.manager import ConfigurationManager
            
            self.log("\nNew Modular Configuration System:")
            new_config = ConfigurationManager.from_environment("development") 
            
            self.log(f"  Environment: {new_config.environment}")
            self.log(f"  Storage Path: {new_config.storage.base_path}")
            self.log(f"  Search Default Limit: {new_config.search.default_limit}")
            self.log(f"  Vector Provider: {new_config.vector.default_embedding_provider}")
            self.log(f"  Document Max Size: {new_config.document.max_file_size_bytes / 1_048_576:.1f}MB")
            self.log(f"  Panel Max Personas: {new_config.panel.max_personas_per_panel}")
            
            # Configuration validation
            errors = new_config.validate_all()
            if errors:
                self.log(f"  Validation Errors: {len(errors)}", "WARN")
                for domain, domain_errors in errors.items():
                    for error in domain_errors[:2]:  # Show first 2 errors
                        self.log(f"    {domain}: {error}")
            else:
                self.log("  ✅ All configurations valid")
                
            # Configuration summary
            summary = new_config.get_summary()
            self.log(f"\n  Configuration Summary:")
            for domain, count in summary.items():
                self.log(f"    {domain}: {count} settings")
                
        except ImportError:
            self.log("\nNew configuration system not available", "WARN")
            
        self.log("Configuration demonstration completed", "SUCCESS")
        
    async def run_performance_benchmark(self):
        """Run basic performance benchmarks."""
        self.separator("PERFORMANCE BENCHMARKS")
        
        # Benchmark message storage
        self.log("Benchmarking message storage performance...")
        start_time = time.time()
        
        benchmark_session_id = await self.storage_manager.create_session(
            user_id="alice_researcher",
            title="Performance Benchmark Session"
        )
        
        # Store multiple messages
        message_count = 50
        for i in range(message_count):
            message = Message(
                role=MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT,
                content=f"This is benchmark message number {i+1}. " * 5  # Make messages longer
            )
            await self.storage_manager.add_message(
                "alice_researcher",
                benchmark_session_id,
                message
            )
            
        storage_time = time.time() - start_time
        self.log(f"  Stored {message_count} messages in {storage_time:.2f}s")
        self.log(f"  Rate: {message_count / storage_time:.1f} messages/second")
        
        # Benchmark search performance
        self.log("\nBenchmarking search performance...")
        start_time = time.time()
        
        search_query = SearchQuery(
            query="benchmark message",
            user_id="alice_researcher",
            session_ids=[benchmark_session_id]
        )
        
        search_results = await self.search_engine.search(search_query)
        search_time = time.time() - start_time
        
        self.log(f"  Searched {message_count} messages in {search_time:.3f}s")
        self.log(f"  Found {len(search_results)} matches")
        self.log(f"  Rate: {message_count / search_time:.0f} messages/second")
        
        self.log("Performance benchmarks completed", "SUCCESS")
        
    async def cleanup_demo_data(self):
        """Clean up demo data with user confirmation."""
        self.separator("DEMO CLEANUP")
        
        if self.verbose:
            print(f"Demo data is stored in: {self.demo_data_path}")
            print(f"Total size: {self.get_directory_size(self.demo_data_path):,} bytes")
            print("\nTo clean up demo data, you can delete the demo_data_automated directory")
            print("or run this script with --cleanup flag")
        
    async def run_complete_demo(self):
        """Run the complete automated demonstration."""
        start_time = time.time()
        
        try:
            await self.initialize()
            await self.create_demo_users()
            await self.create_demo_sessions()
            await self.populate_chat_messages()
            await self.create_demo_documents()
            await self.create_vector_embeddings()
            await self.demonstrate_search_capabilities()
            await self.demonstrate_configuration_systems()
            await self.run_performance_benchmark()
            await self.show_system_statistics()
            
            total_time = time.time() - start_time
            
            self.separator("DEMO COMPLETED SUCCESSFULLY")
            self.log(f"🎉 Complete demo finished in {total_time:.1f} seconds!")
            self.log(f"📊 Created {len(self.users)} users, {len(self.sessions)} sessions")
            self.log(f"📝 Stored {len(self.messages)} messages, {len(self.documents)} documents")
            self.log(f"💾 Demo data saved to: {self.demo_data_path}")
            
            await self.cleanup_demo_data()
            
        except Exception as e:
            self.log(f"Demo failed with error: {e}", "ERROR")
            raise


def create_mock_embedding(text: str, dim: int = 384) -> list:
    """Create reproducible mock embedding based on text hash."""
    hash_val = hash(text) % (2**31)
    np.random.seed(hash_val)
    return np.random.normal(0, 1, dim).tolist()


async def main():
    """Main entry point for the automated demo."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Flatfile Chat Database Automated Demo")
    parser.add_argument("--quiet", action="store_true", help="Run in quiet mode")
    parser.add_argument("--cleanup", action="store_true", help="Clean up demo data after completion")
    
    args = parser.parse_args()
    
    print("🚀 Starting Flatfile Chat Database Automated Demo")
    print("=" * 60)
    print()
    print("This comprehensive demo will showcase all major features:")
    print("• User management and profiles")
    print("• Chat session creation and message storage")
    print("• Document processing and storage")
    print("• Vector embeddings and similarity search")
    print("• Text-based search capabilities")
    print("• Configuration systems (legacy and new)")
    print("• Performance benchmarks")
    print("• System statistics and file structure")
    print()
    
    if not args.quiet:
        input("Press Enter to begin the demo...")
        print()
    
    demo = AutomatedDemo(verbose=not args.quiet)
    await demo.run_complete_demo()
    
    if args.cleanup:
        import shutil
        shutil.rmtree(demo.demo_data_path, ignore_errors=True)
        print(f"\n🧹 Cleaned up demo data from {demo.demo_data_path}")


if __name__ == "__main__":
    asyncio.run(main())