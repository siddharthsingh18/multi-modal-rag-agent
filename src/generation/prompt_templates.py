"""Prompt templates for RAG system."""

from typing import List, Optional


class PromptTemplates:
    """Collection of prompt templates for different tasks."""

    @staticmethod
    def rag_prompt(
        query: str,
        context: List[str],
        include_sources: bool = True,
    ) -> str:
        """
        Generate RAG prompt with context.

        Args:
            query: User query
            context: List of retrieved context chunks
            include_sources: Whether to ask for source citations

        Returns:
            Formatted prompt
        """
        context_text = "\n\n".join(
            [f"[Document {i+1}]\n{chunk}" for i, chunk in enumerate(context)]
        )

        citation_instruction = (
            "\n\nPlease cite the document numbers you use in your answer (e.g., [1], [2])."
            if include_sources
            else ""
        )

        return f"""You are a helpful AI assistant. Use the following context to answer the user's question.
If the context doesn't contain relevant information, say so clearly.

Context:
{context_text}

Question: {query}{citation_instruction}

Answer:"""

    @staticmethod
    def system_prompt() -> str:
        """Get default system prompt."""
        return """You are a knowledgeable AI assistant with expertise in retrieving and synthesizing information from documents.
Your responses should be:
- Accurate and based on the provided context
- Clear and well-structured
- Honest when information is not available
- Professional and helpful"""

    @staticmethod
    def query_rewrite_prompt(query: str) -> str:
        """
        Generate prompt for query rewriting.

        Args:
            query: Original query

        Returns:
            Query rewriting prompt
        """
        return f"""Rewrite the following query to make it more specific and suitable for document retrieval.
Generate 2-3 alternative phrasings that capture the same intent.

Original query: {query}

Rewritten queries (one per line):"""

    @staticmethod
    def multi_query_prompt(query: str) -> str:
        """
        Generate multiple search queries from a single question.

        Args:
            query: Original query

        Returns:
            Multi-query generation prompt
        """
        return f"""Generate 3 different search queries that would help answer the following question from different angles.
Each query should focus on a different aspect or perspective.

Question: {query}

Search queries (one per line):"""

    @staticmethod
    def reflection_prompt(
        query: str,
        answer: str,
        context: List[str],
    ) -> str:
        """
        Generate reflection prompt for answer quality.

        Args:
            query: Original query
            answer: Generated answer
            context: Context used

        Returns:
            Reflection prompt
        """
        context_text = "\n".join(context)

        return f"""Evaluate the quality of the following answer given the question and context.

Question: {query}

Context:
{context_text}

Answer:
{answer}

Please evaluate:
1. Does the answer accurately address the question?
2. Is the answer well-supported by the context?
3. Are there any inconsistencies or hallucinations?
4. What could be improved?

Provide a brief evaluation and a quality score from 1-10:"""

    @staticmethod
    def summarization_prompt(text: str, max_words: int = 100) -> str:
        """
        Generate summarization prompt.

        Args:
            text: Text to summarize
            max_words: Maximum words in summary

        Returns:
            Summarization prompt
        """
        return f"""Provide a concise summary of the following text in no more than {max_words} words.
Focus on the key points and main ideas.

Text:
{text}

Summary:"""

    @staticmethod
    def extraction_prompt(
        text: str,
        fields: List[str],
    ) -> str:
        """
        Generate information extraction prompt.

        Args:
            text: Text to extract from
            fields: List of fields to extract

        Returns:
            Extraction prompt
        """
        fields_text = "\n".join([f"- {field}" for field in fields])

        return f"""Extract the following information from the text:

{fields_text}

Text:
{text}

Extracted information (JSON format):"""

    @staticmethod
    def agent_planning_prompt(task: str, available_tools: List[str]) -> str:
        """
        Generate agent planning prompt.

        Args:
            task: Task to plan
            available_tools: List of available tools

        Returns:
            Planning prompt
        """
        tools_text = "\n".join([f"- {tool}" for tool in available_tools])

        return f"""You are a planning agent. Break down the following task into concrete steps.

Available tools:
{tools_text}

Task: {task}

Provide a step-by-step plan to accomplish this task:"""

    @staticmethod
    def code_analysis_prompt(code: str, question: str) -> str:
        """
        Generate code analysis prompt.

        Args:
            code: Code to analyze
            question: Question about the code

        Returns:
            Code analysis prompt
        """
        return f"""Analyze the following code and answer the question.

Code:
```
{code}
```

Question: {question}

Analysis:"""


# Convenience functions
def format_rag_prompt(query: str, context: List[str]) -> str:
    """Format RAG prompt with query and context."""
    return PromptTemplates.rag_prompt(query, context)


def get_system_prompt() -> str:
    """Get default system prompt."""
    return PromptTemplates.system_prompt()
