"""Pytest fixtures for the Type 4 defense pipeline."""

from typing import Dict, List, Tuple

import pytest

from .dataset_helpers import (
    ChunkInfo,
    Document,
    N_AGENT_POISON,
    N_BENIGN,
    N_POISONED_RAG,
    N_SEMANTIC_CONFUSION,
    generate_agentpoison_embeddings,
    generate_benign_embeddings,
    generate_poisonedrag_embeddings,
    get_all_attack_labels,
    get_all_benign_labels,
    make_agent_hijacking_docs,
    make_agentpoison_docs,
    make_benign_chunk_groups,
    make_benign_docs,
    make_enhanced_agentpoison_docs,
    make_enhanced_poisonedrag_docs,
    make_enhanced_semantic_confusion_groups,
    make_extra_benign_docs,
    make_hybrid_attack_docs,
    make_memory_poisoning_docs,
    make_poisonedrag_docs,
    make_prompt_injection_docs,
    make_semantic_confusion_groups,
    make_tool_misuse_docs,
    mock_cross_encoder,
    mock_embed_model,
    mock_llm_generate,
    mock_nli_model,
)


@pytest.fixture(scope="session")
def benign_docs() -> List[Document]:
    return make_benign_docs(N_BENIGN)


@pytest.fixture(scope="session")
def poisonedrag_docs() -> List[Document]:
    return make_poisonedrag_docs(N_POISONED_RAG)


@pytest.fixture(scope="session")
def agentpoison_docs() -> List[Document]:
    return make_agentpoison_docs(N_AGENT_POISON)


@pytest.fixture(scope="session")
def benign_embeddings() -> List[List[float]]:
    return generate_benign_embeddings(N_BENIGN)


@pytest.fixture(scope="session")
def poisonedrag_embeddings() -> List[List[float]]:
    return generate_poisonedrag_embeddings(N_POISONED_RAG)


@pytest.fixture(scope="session")
def agentpoison_embeddings() -> List[List[float]]:
    """AgentPoison 紧凑嵌入（20 个围绕一个中心）。"""
    return generate_agentpoison_embeddings(N_AGENT_POISON)


@pytest.fixture(scope="session")
def combined_embeddings(benign_embeddings, poisonedrag_embeddings, agentpoison_embeddings):
    """混合嵌入集：40 良性 + 20 PoisonedRAG + 20 AgentPoison。"""
    return benign_embeddings + poisonedrag_embeddings + agentpoison_embeddings


@pytest.fixture(scope="session")
def combined_doc_embeddings(benign_docs, benign_embeddings,
                             poisonedrag_docs, poisonedrag_embeddings) -> List:
    """用于 SP1 的 DocumentEmbedding 列表：带 doc_id 的嵌入。"""
    from External_Developer_Write.types import DocumentEmbedding
    embeds = []
    for doc, vec in zip(benign_docs[:20], benign_embeddings[:20]):
        embeds.append(DocumentEmbedding(doc_id=doc.doc_id, vector=vec))
    for doc, vec in zip(poisonedrag_docs[:10], poisonedrag_embeddings[:10]):
        embeds.append(DocumentEmbedding(doc_id=doc.doc_id, vector=vec))
    return embeds


@pytest.fixture(scope="session")
def semantic_confusion_groups() -> List[Tuple[str, List[ChunkInfo]]]:
    return make_semantic_confusion_groups(N_SEMANTIC_CONFUSION)


@pytest.fixture(scope="session")
def benign_chunk_groups() -> List[Tuple[str, List[ChunkInfo]]]:
    return make_benign_chunk_groups(N_BENIGN)


@pytest.fixture(scope="session")
def mock_models():
    return {
        "embed": mock_embed_model,
        "nli": mock_nli_model,
        "cross_encoder": mock_cross_encoder,
        "llm": mock_llm_generate,
    }


@pytest.fixture(scope="session")
def extra_benign_docs() -> List[Document]:
    return make_extra_benign_docs()


@pytest.fixture(scope="session")
def prompt_injection_docs() -> List[Document]:
    return make_prompt_injection_docs()


@pytest.fixture(scope="session")
def tool_misuse_docs() -> List[Document]:
    return make_tool_misuse_docs()


@pytest.fixture(scope="session")
def memory_poisoning_docs() -> List[Document]:
    return make_memory_poisoning_docs()


@pytest.fixture(scope="session")
def agent_hijacking_docs() -> List[Document]:
    return make_agent_hijacking_docs()


@pytest.fixture(scope="session")
def enhanced_poisonedrag_docs() -> List[Document]:
    return make_enhanced_poisonedrag_docs()


@pytest.fixture(scope="session")
def enhanced_agentpoison_docs() -> List[Document]:
    return make_enhanced_agentpoison_docs()


@pytest.fixture(scope="session")
def enhanced_semantic_confusion_groups() -> List[Tuple[str, List[ChunkInfo]]]:
    return make_enhanced_semantic_confusion_groups(10)


@pytest.fixture(scope="session")
def hybrid_attack_docs() -> List[Document]:
    return make_hybrid_attack_docs()


@pytest.fixture(scope="session")
def all_attack_labels() -> Dict[str, str]:
    return get_all_attack_labels()


@pytest.fixture(scope="session")
def all_benign_labels() -> Dict[str, str]:
    return get_all_benign_labels()
