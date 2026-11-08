"""A configured semantic extractor returns triples with one document-ID wrapper."""
from External_Developer_Write.sp7_semantic_graph import SemanticDependencyGraphAnalyzer
from External_Developer_Write.types import Document, SemanticTriple


def test_real_extractor_triples_are_not_double_wrapped():
    analyzer=SemanticDependencyGraphAnalyzer(
        nlp_extractor=lambda text,k:[SemanticTriple("Team","uses","Database"),
                                     SemanticTriple("Team","maintains","Docs")])
    result=analyzer.analyze([Document("a","Team database docs"),Document("b","Team docs")])
    assert result.details["entity_count"]==3
