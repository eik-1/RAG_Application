import json
import asyncio
import time
from typing import Dict, List, Any, Tuple
from pathlib import Path
from datetime import datetime
import pandas as pd
from loguru import logger

# Import RAGAS for evaluation (if available)
try:
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    HAS_RAGAS = True
except ImportError:
    logger.warning("RAGAS not available, using custom evaluation metrics")
    HAS_RAGAS = False

# For text similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

from app.rag_system import rag_manager
from config import settings


class CustomEvaluationMetrics:
    """Custom evaluation metrics for RAG responses."""
    
    def __init__(self):
        self.stemmer = PorterStemmer()
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for evaluation."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and extra whitespace
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Tokenize and remove stopwords
        try:
            words = word_tokenize(text)
        except:
            words = text.split()
        
        words = [self.stemmer.stem(word) for word in words if word not in self.stop_words]
        
        return ' '.join(words)
    
    def relevance_score(self, question: str, answer: str, sources: List[Dict[str, Any]]) -> float:
        """Calculate relevance of answer to question."""
        
        # Preprocess texts
        q_processed = self.preprocess_text(question)
        a_processed = self.preprocess_text(answer)
        
        # Calculate TF-IDF similarity
        vectorizer = TfidfVectorizer()
        try:
            tfidf_matrix = vectorizer.fit_transform([q_processed, a_processed])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        except:
            # Fallback to simple word overlap
            q_words = set(q_processed.split())
            a_words = set(a_processed.split())
            if len(q_words) == 0 or len(a_words) == 0:
                return 0.0
            similarity = len(q_words.intersection(a_words)) / len(q_words.union(a_words))
        
        # Boost score if sources are highly relevant
        if sources:
            avg_source_score = sum(source.get('similarity_score', 0) for source in sources) / len(sources)
            similarity = (similarity + avg_source_score) / 2
        
        return min(similarity, 1.0)
    
    def accuracy_score(self, answer: str, expected_topics: List[str]) -> float:
        """Calculate accuracy based on presence of expected topics."""
        
        answer_processed = self.preprocess_text(answer)
        
        topic_mentions = 0
        for topic in expected_topics:
            topic_processed = self.preprocess_text(topic)
            
            # Check for exact matches or partial matches
            if topic_processed in answer_processed:
                topic_mentions += 1
            else:
                # Check for partial matches (individual words)
                topic_words = topic_processed.split()
                if any(word in answer_processed for word in topic_words if len(word) > 3):
                    topic_mentions += 0.5
        
        return min(topic_mentions / len(expected_topics) if expected_topics else 0, 1.0)
    
    def coherence_score(self, answer: str) -> float:
        """Calculate coherence of the answer."""
        
        if not answer or len(answer.strip()) < 10:
            return 0.0
        
        sentences = answer.split('.')
        valid_sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        
        if len(valid_sentences) < 2:
            return 0.5  # Short but potentially coherent
        
        # Simple coherence checks
        score = 1.0
        
        # Check for repetition
        if len(set(valid_sentences)) / len(valid_sentences) < 0.7:
            score -= 0.2
        
        # Check for very short sentences (might indicate fragmentation)
        avg_sentence_length = sum(len(s.split()) for s in valid_sentences) / len(valid_sentences)
        if avg_sentence_length < 5:
            score -= 0.2
        
        # Check for very long sentences (might indicate run-on)
        if any(len(s.split()) > 50 for s in valid_sentences):
            score -= 0.1
        
        return max(score, 0.0)
    
    def faithfulness_score(self, answer: str, sources: List[Dict[str, Any]]) -> float:
        """Calculate faithfulness of answer to sources."""
        
        if not sources:
            return 0.5  # Neutral if no sources
        
        answer_processed = self.preprocess_text(answer)
        source_texts = [self.preprocess_text(source.get('snippet', '')) for source in sources]
        
        if not answer_processed or not any(source_texts):
            return 0.0
        
        # Calculate overlap with sources
        answer_words = set(answer_processed.split())
        source_words = set()
        for source_text in source_texts:
            source_words.update(source_text.split())
        
        if not answer_words or not source_words:
            return 0.0
        
        overlap = len(answer_words.intersection(source_words))
        faithfulness = overlap / len(answer_words)
        
        return min(faithfulness, 1.0)


class RAGEvaluationFramework:
    """Comprehensive evaluation framework for RAG systems."""
    
    def __init__(self):
        self.metrics = CustomEvaluationMetrics()
        self.test_questions_file = Path(settings.test_questions_file)
        self.results_dir = Path(settings.evaluation_output_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def load_test_questions(self) -> List[Dict[str, Any]]:
        """Load test questions from JSON file."""
        
        if not self.test_questions_file.exists():
            raise FileNotFoundError(f"Test questions file not found: {self.test_questions_file}")
        
        with open(self.test_questions_file, 'r', encoding='utf-8') as f:
            questions = json.load(f)
        
        logger.info(f"Loaded {len(questions)} test questions")
        return questions
    
    async def evaluate_single_question(
        self, 
        question_data: Dict[str, Any],
        rag_system,
        context_questions: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Evaluate a single question."""
        
        question = question_data['question']
        expected_topics = question_data.get('expected_topics', [])
        needs_context = question_data.get('context_memory', False)
        
        logger.info(f"Evaluating question {question_data['id']}: {question[:50]}...")
        
        # If this question needs context, ask context questions first
        if needs_context and context_questions:
            for ctx_q in context_questions:
                await rag_system.query(ctx_q['question'])
                await asyncio.sleep(0.5)  # Small delay
        
        # Measure response time
        start_time = time.time()
        
        try:
            # Get RAG response
            response = await rag_system.query(question)
            response_time = time.time() - start_time
            
            answer = response.get('response', '')
            sources = response.get('sources', [])
            retrieval_stats = response.get('retrieval_stats', {})
            
            # Calculate metrics
            relevance = self.metrics.relevance_score(question, answer, sources)
            accuracy = self.metrics.accuracy_score(answer, expected_topics)
            coherence = self.metrics.coherence_score(answer)
            faithfulness = self.metrics.faithfulness_score(answer, sources)
            
            # Overall score (weighted average)
            overall_score = (
                relevance * 0.3 +
                accuracy * 0.3 +
                coherence * 0.2 +
                faithfulness * 0.2
            )
            
            result = {
                'question_id': question_data['id'],
                'question': question,
                'category': question_data.get('category', 'Unknown'),
                'difficulty': question_data.get('difficulty', 'Unknown'),
                'answer': answer,
                'response_time': response_time,
                'sources_count': len(sources),
                'top_similarity_score': retrieval_stats.get('top_score', 0),
                'chunks_retrieved': retrieval_stats.get('chunks_found', 0),
                'metrics': {
                    'relevance': relevance,
                    'accuracy': accuracy,
                    'coherence': coherence,
                    'faithfulness': faithfulness,
                    'overall_score': overall_score
                },
                'sources': sources[:3],  # Keep top 3 sources for analysis
                'expected_topics': expected_topics,
                'topics_found': self._analyze_topics_coverage(answer, expected_topics)
            }
            
            logger.success(f"Question {question_data['id']} evaluated - Score: {overall_score:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to evaluate question {question_data['id']}: {str(e)}")
            return {
                'question_id': question_data['id'],
                'question': question,
                'error': str(e),
                'metrics': {
                    'relevance': 0,
                    'accuracy': 0,
                    'coherence': 0,
                    'faithfulness': 0,
                    'overall_score': 0
                }
            }
    
    def _analyze_topics_coverage(self, answer: str, expected_topics: List[str]) -> List[str]:
        """Analyze which expected topics are covered in the answer."""
        
        answer_lower = answer.lower()
        found_topics = []
        
        for topic in expected_topics:
            topic_lower = topic.lower()
            if topic_lower in answer_lower:
                found_topics.append(topic)
            else:
                # Check for partial matches
                topic_words = topic_lower.split()
                if any(word in answer_lower for word in topic_words if len(word) > 3):
                    found_topics.append(f"{topic} (partial)")
        
        return found_topics
    
    async def run_evaluation(self, clear_memory_between: bool = True) -> Dict[str, Any]:
        """Run complete evaluation on all test questions."""
        
        logger.info("Starting RAG system evaluation...")
        
        # Load test questions
        questions = self.load_test_questions()
        
        # Initialize RAG system
        rag_system = rag_manager.get_rag_system()
        if not rag_system.is_initialized:
            logger.info("Initializing RAG system for evaluation...")
            await rag_system.initialize()
        
        # Prepare context questions for questions that need memory
        context_questions = [q for q in questions if not q.get('context_memory', False)][:4]
        
        # Run evaluation
        results = []
        start_time = time.time()
        
        for i, question_data in enumerate(questions):
            if clear_memory_between and i > 0:
                rag_system.clear_conversation_memory()
            
            result = await self.evaluate_single_question(
                question_data, 
                rag_system,
                context_questions if question_data.get('context_memory', False) else None
            )
            results.append(result)
            
            # Small delay between questions
            await asyncio.sleep(1)
        
        total_time = time.time() - start_time
        
        # Calculate summary statistics
        summary = self._calculate_summary_stats(results, total_time)
        
        # Save results
        evaluation_data = {
            'timestamp': datetime.now().isoformat(),
            'summary': summary,
            'detailed_results': results,
            'configuration': {
                'total_questions': len(questions),
                'clear_memory_between': clear_memory_between,
                'rag_system_config': rag_system.get_system_status()
            }
        }
        
        # Save to file
        output_file = self.results_dir / f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation_data, f, indent=2, ensure_ascii=False)
        
        logger.success(f"Evaluation completed! Results saved to {output_file}")
        return evaluation_data
    
    def _calculate_summary_stats(self, results: List[Dict[str, Any]], total_time: float) -> Dict[str, Any]:
        """Calculate summary statistics from evaluation results."""
        
        valid_results = [r for r in results if 'error' not in r]
        
        if not valid_results:
            return {'error': 'No valid results to analyze'}
        
        # Extract metrics
        metrics_data = []
        for result in valid_results:
            metrics = result.get('metrics', {})
            metrics_data.append({
                'relevance': metrics.get('relevance', 0),
                'accuracy': metrics.get('accuracy', 0),
                'coherence': metrics.get('coherence', 0),
                'faithfulness': metrics.get('faithfulness', 0),
                'overall_score': metrics.get('overall_score', 0),
                'response_time': result.get('response_time', 0),
                'sources_count': result.get('sources_count', 0),
                'chunks_retrieved': result.get('chunks_retrieved', 0),
                'category': result.get('category', 'Unknown'),
                'difficulty': result.get('difficulty', 'Unknown')
            })
        
        df = pd.DataFrame(metrics_data)
        
        # Calculate statistics
        summary = {
            'total_questions': len(results),
            'successful_evaluations': len(valid_results),
            'failed_evaluations': len(results) - len(valid_results),
            'total_evaluation_time': total_time,
            'avg_response_time': df['response_time'].mean(),
            
            # Metric scores
            'overall_metrics': {
                'relevance': {
                    'mean': df['relevance'].mean(),
                    'std': df['relevance'].std(),
                    'min': df['relevance'].min(),
                    'max': df['relevance'].max()
                },
                'accuracy': {
                    'mean': df['accuracy'].mean(),
                    'std': df['accuracy'].std(),
                    'min': df['accuracy'].min(),
                    'max': df['accuracy'].max()
                },
                'coherence': {
                    'mean': df['coherence'].mean(),
                    'std': df['coherence'].std(),
                    'min': df['coherence'].min(),
                    'max': df['coherence'].max()
                },
                'faithfulness': {
                    'mean': df['faithfulness'].mean(),
                    'std': df['faithfulness'].std(),
                    'min': df['faithfulness'].min(),
                    'max': df['faithfulness'].max()
                },
                'overall_score': {
                    'mean': df['overall_score'].mean(),
                    'std': df['overall_score'].std(),
                    'min': df['overall_score'].min(),
                    'max': df['overall_score'].max()
                }
            },
            
            # Category analysis
            'category_performance': df.groupby('category')['overall_score'].agg(['mean', 'count']).to_dict(),
            'difficulty_performance': df.groupby('difficulty')['overall_score'].agg(['mean', 'count']).to_dict(),
            
            # Retrieval statistics
            'retrieval_stats': {
                'avg_sources_per_question': df['sources_count'].mean(),
                'avg_chunks_retrieved': df['chunks_retrieved'].mean(),
            }
        }
        
        return summary
    
    def generate_report(self, evaluation_data: Dict[str, Any]) -> str:
        """Generate a human-readable evaluation report."""
        
        summary = evaluation_data['summary']
        
        report_lines = [
            "# RAG System Evaluation Report",
            f"**Evaluation Date:** {evaluation_data['timestamp']}",
            "",
            "## Summary",
            f"- **Total Questions:** {summary['total_questions']}",
            f"- **Successful Evaluations:** {summary['successful_evaluations']}",
            f"- **Failed Evaluations:** {summary['failed_evaluations']}",
            f"- **Total Evaluation Time:** {summary['total_evaluation_time']:.2f} seconds",
            f"- **Average Response Time:** {summary['avg_response_time']:.2f} seconds",
            "",
            "## Overall Performance",
        ]
        
        metrics = summary['overall_metrics']
        for metric_name, metric_data in metrics.items():
            report_lines.extend([
                f"### {metric_name.title()}",
                f"- **Mean:** {metric_data['mean']:.3f}",
                f"- **Standard Deviation:** {metric_data['std']:.3f}",
                f"- **Range:** {metric_data['min']:.3f} - {metric_data['max']:.3f}",
                ""
            ])
        
        # Add category and difficulty analysis
        report_lines.extend([
            "## Performance by Category",
            ""
        ])
        
        for category, performance in summary['category_performance'].items():
            report_lines.append(f"- **{category}:** {performance['mean']:.3f} (n={performance['count']})")
        
        report_lines.extend([
            "",
            "## Performance by Difficulty",
            ""
        ])
        
        for difficulty, performance in summary['difficulty_performance'].items():
            report_lines.append(f"- **{difficulty}:** {performance['mean']:.3f} (n={performance['count']})")
        
        return "\n".join(report_lines) 