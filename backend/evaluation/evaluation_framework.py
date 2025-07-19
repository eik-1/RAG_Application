"""Evaluation framework for the RAG system."""

import json
import time
from typing import List, Dict, Any
from pathlib import Path
import nltk
from rouge_score import rouge_scorer
import re

from config import TEST_QUESTIONS_FILE, EVALUATION_RESULTS_FILE


class EvaluationFramework:
    """Comprehensive evaluation framework for RAG system."""
    
    def __init__(self, rag_system):
        self.rag_system = rag_system
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        
        # Download NLTK data if needed
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            print("Downloading NLTK punkt tokenizer...")
            nltk.download('punkt', quiet=True)
    
    def load_test_questions(self) -> List[Dict[str, Any]]:
        """Load test questions from JSON file."""
        if not TEST_QUESTIONS_FILE.exists():
            raise FileNotFoundError(f"Test questions file not found: {TEST_QUESTIONS_FILE}")
        
        with open(TEST_QUESTIONS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def evaluate_single_response(self, question: str, response: str, 
                               sources: List[Dict[str, Any]], 
                               expected_sources: List[str] = None) -> Dict[str, Any]:
        """Evaluate a single response using multiple metrics."""
        
        # 1. Relevance Score (based on keyword matching)
        relevance_score = self._calculate_relevance(question, response)
        
        # 2. Coherence Score (based on sentence structure and flow)
        coherence_score = self._calculate_coherence(response)
        
        # 3. Faithfulness Score (response should be grounded in sources)
        faithfulness_score = self._calculate_faithfulness(response, sources)
        
        # 4. Source Coverage (checking if expected sources are retrieved)
        source_coverage = self._calculate_source_coverage(sources, expected_sources)
        
        # 5. Response Quality (length, completeness, clarity)
        quality_score = self._calculate_quality(response)
        
        # 6. Overall Score (weighted average)
        overall_score = (
            relevance_score * 0.25 +
            coherence_score * 0.20 +
            faithfulness_score * 0.25 +
            source_coverage * 0.15 +
            quality_score * 0.15
        )
        
        return {
            "relevance_score": relevance_score,
            "coherence_score": coherence_score,
            "faithfulness_score": faithfulness_score,
            "source_coverage": source_coverage,
            "quality_score": quality_score,
            "overall_score": overall_score,
            "response_length": len(response),
            "num_sources": len(sources)
        }
    
    def _calculate_relevance(self, question: str, response: str) -> float:
        """Calculate relevance between question and response."""
        question_words = set(re.findall(r'\w+', question.lower()))
        response_words = set(re.findall(r'\w+', response.lower()))
        
        # Remove common stop words
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
        question_words -= stop_words
        response_words -= stop_words
        
        if not question_words:
            return 0.0
        
        overlap = len(question_words.intersection(response_words))
        return min(overlap / len(question_words), 1.0)
    
    def _calculate_coherence(self, response: str) -> float:
        """Calculate coherence based on sentence structure."""
        sentences = nltk.sent_tokenize(response)
        
        if len(sentences) == 0:
            return 0.0
        
        # Check for proper sentence structure
        coherence_indicators = 0
        total_checks = 0
        
        for sentence in sentences:
            total_checks += 1
            
            # Check if sentence has reasonable length
            if 10 <= len(sentence.split()) <= 50:
                coherence_indicators += 1
            
            # Check if sentence starts with capital letter
            if sentence.strip() and sentence.strip()[0].isupper():
                coherence_indicators += 1
                total_checks += 1
            
            # Check if sentence ends with proper punctuation
            if sentence.strip() and sentence.strip()[-1] in '.!?':
                coherence_indicators += 1
                total_checks += 1
        
        return coherence_indicators / total_checks if total_checks > 0 else 0.0
    
    def _calculate_faithfulness(self, response: str, sources: List[Dict[str, Any]]) -> float:
        """Calculate how well the response is grounded in sources."""
        if not sources:
            return 0.0
        
        response_words = set(re.findall(r'\w+', response.lower()))
        source_words = set()
        
        for source in sources:
            text = source.get('text', '') + source.get('text_preview', '')
            source_words.update(re.findall(r'\w+', text.lower()))
        
        if not response_words:
            return 0.0
        
        # Calculate overlap between response and source content
        overlap = len(response_words.intersection(source_words))
        return min(overlap / len(response_words), 1.0)
    
    def _calculate_source_coverage(self, sources: List[Dict[str, Any]], 
                                 expected_sources: List[str] = None) -> float:
        """Calculate how well the retrieved sources match expected sources."""
        if not expected_sources:
            return 1.0  # No specific expectation
        
        retrieved_sources = set()
        for source in sources:
            source_name = source.get('source', '')
            retrieved_sources.add(source_name)
        
        expected_set = set(expected_sources)
        overlap = len(retrieved_sources.intersection(expected_set))
        
        return overlap / len(expected_set) if expected_set else 1.0
    
    def _calculate_quality(self, response: str) -> float:
        """Calculate overall quality of the response."""
        if not response or len(response.strip()) < 10:
            return 0.0
        
        quality_score = 0.0
        
        # Length check (not too short, not too long)
        length = len(response)
        if 50 <= length <= 1000:
            quality_score += 0.3
        elif length > 20:
            quality_score += 0.15
        
        # Check for informative content
        if any(keyword in response.lower() for keyword in ['transformer', 'attention', 'model', 'training', 'bert', 'gpt']):
            quality_score += 0.3
        
        # Check for explanation indicators
        if any(phrase in response.lower() for phrase in ['because', 'due to', 'however', 'therefore', 'this means']):
            quality_score += 0.2
        
        # Check for technical depth
        if any(term in response.lower() for term in ['architecture', 'mechanism', 'algorithm', 'approach', 'method']):
            quality_score += 0.2
        
        return min(quality_score, 1.0)
    
    def run_evaluation(self, quick_mode: bool = False) -> Dict[str, Any]:
        """Run complete evaluation on all test questions."""
        print("🔍 Starting RAG system evaluation...")
        start_time = time.time()
        
        # Load test questions
        test_questions = self.load_test_questions()
        
        if quick_mode:
            test_questions = test_questions[:3]  # Only first 3 questions for quick test
            print(f"Running quick evaluation with {len(test_questions)} questions...")
        else:
            print(f"Running full evaluation with {len(test_questions)} questions...")
        
        results = []
        category_scores = {}
        
        for i, question_data in enumerate(test_questions, 1):
            print(f"📝 Evaluating question {i}/{len(test_questions)}: {question_data['question'][:80]}...")
            
            # Get response from RAG system
            rag_response = self.rag_system.chat(question_data['question'])
            
            if not rag_response['success']:
                print(f"❌ Failed to get response: {rag_response.get('error', 'Unknown error')}")
                continue
            
            # Evaluate the response
            evaluation = self.evaluate_single_response(
                question=question_data['question'],
                response=rag_response['response'],
                sources=rag_response['sources'],
                expected_sources=question_data.get('expected_sources', [])
            )
            
            # Store result
            result = {
                "question_id": question_data['id'],
                "question": question_data['question'],
                "category": question_data['category'],
                "difficulty": question_data['difficulty'],
                "response": rag_response['response'],
                "sources": rag_response['sources'],
                "evaluation": evaluation,
                "processing_time": rag_response.get('processing_time', 0)
            }
            results.append(result)
            
            # Track category performance
            category = question_data['category']
            if category not in category_scores:
                category_scores[category] = []
            category_scores[category].append(evaluation['overall_score'])
            
            print(f"✅ Overall score: {evaluation['overall_score']:.3f}")
        
        # Calculate aggregate statistics
        total_time = time.time() - start_time
        
        if results:
            overall_scores = [r['evaluation']['overall_score'] for r in results]
            relevance_scores = [r['evaluation']['relevance_score'] for r in results]
            coherence_scores = [r['evaluation']['coherence_score'] for r in results]
            faithfulness_scores = [r['evaluation']['faithfulness_score'] for r in results]
            
            aggregate_stats = {
                "total_questions": len(results),
                "average_overall_score": sum(overall_scores) / len(overall_scores),
                "average_relevance": sum(relevance_scores) / len(relevance_scores),
                "average_coherence": sum(coherence_scores) / len(coherence_scores),
                "average_faithfulness": sum(faithfulness_scores) / len(faithfulness_scores),
                "total_evaluation_time": total_time,
                "average_time_per_question": total_time / len(results)
            }
            
            # Category breakdown
            category_breakdown = {}
            for category, scores in category_scores.items():
                category_breakdown[category] = {
                    "count": len(scores),
                    "average_score": sum(scores) / len(scores),
                    "max_score": max(scores),
                    "min_score": min(scores)
                }
        else:
            aggregate_stats = {"error": "No successful evaluations"}
            category_breakdown = {}
        
        # Prepare final results
        evaluation_results = {
            "timestamp": time.time(),
            "quick_mode": quick_mode,
            "aggregate_statistics": aggregate_stats,
            "category_breakdown": category_breakdown,
            "detailed_results": results,
            "system_stats": self.rag_system.get_system_stats()
        }
        
        # Save results
        self.save_results(evaluation_results)
        
        print(f"📊 Evaluation completed in {total_time:.2f} seconds")
        if results:
            print(f"📈 Average overall score: {aggregate_stats['average_overall_score']:.3f}")
        
        return evaluation_results
    
    def save_results(self, results: Dict[str, Any]):
        """Save evaluation results to file."""
        try:
            with open(EVALUATION_RESULTS_FILE, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"💾 Results saved to {EVALUATION_RESULTS_FILE}")
        except Exception as e:
            print(f"❌ Failed to save results: {e}")
    
    def generate_report(self, results: Dict[str, Any] = None) -> str:
        """Generate a human-readable evaluation report."""
        if results is None:
            # Load latest results
            if EVALUATION_RESULTS_FILE.exists():
                with open(EVALUATION_RESULTS_FILE, 'r', encoding='utf-8') as f:
                    results = json.load(f)
            else:
                return "No evaluation results found."
        
        report = []
        report.append("=" * 60)
        report.append("RAG SYSTEM EVALUATION REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Summary
        stats = results.get("aggregate_statistics", {})
        if "average_overall_score" in stats:
            report.append(f"📊 SUMMARY")
            report.append(f"   Total Questions: {stats['total_questions']}")
            report.append(f"   Average Score: {stats['average_overall_score']:.3f}/1.0")
            report.append(f"   Average Relevance: {stats['average_relevance']:.3f}/1.0")
            report.append(f"   Average Coherence: {stats['average_coherence']:.3f}/1.0")
            report.append(f"   Average Faithfulness: {stats['average_faithfulness']:.3f}/1.0")
            report.append(f"   Total Time: {stats['total_evaluation_time']:.2f}s")
            report.append("")
        
        # Category breakdown
        categories = results.get("category_breakdown", {})
        if categories:
            report.append("📈 PERFORMANCE BY CATEGORY")
            for category, data in categories.items():
                report.append(f"   {category}: {data['average_score']:.3f} (n={data['count']})")
            report.append("")
        
        # System info
        system_stats = results.get("system_stats", {})
        if system_stats:
            report.append("🔧 SYSTEM CONFIGURATION")
            report.append(f"   Total Chunks: {system_stats.get('total_chunks', 'N/A')}")
            report.append(f"   Vector Dimension: {system_stats.get('vector_dimension', 'N/A')}")
            report.append(f"   Embedding Model: {system_stats.get('embedding_model', 'N/A')}")
            report.append("")
        
        return "\n".join(report) 