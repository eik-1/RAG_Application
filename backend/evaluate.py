#!/usr/bin/env python3
"""
Main evaluation script for the RAG Application.

This script runs the comprehensive evaluation framework on the RAG system,
testing it with predefined questions and generating detailed metrics.

Usage:
    python evaluate.py [--clear-memory] [--output-dir DIR]
"""

import asyncio
import argparse
import sys
from pathlib import Path
from loguru import logger

# Add the backend directory to the Python path
sys.path.append(str(Path(__file__).parent))

from evaluation.evaluation_framework import RAGEvaluationFramework
from app.rag_system import rag_manager


async def main():
    """Main evaluation function."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Evaluate the RAG Application with comprehensive metrics"
    )
    parser.add_argument(
        '--clear-memory',
        action='store_true',
        help='Clear conversation memory between questions'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory to save evaluation results'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger.remove()
    logger.add(sys.stderr, level=log_level, format="{time} | {level} | {message}")
    
    logger.info("🚀 Starting RAG Application Evaluation")
    logger.info("=" * 50)
    
    try:
        # Initialize evaluation framework
        evaluator = RAGEvaluationFramework()
        
        if args.output_dir:
            evaluator.results_dir = Path(args.output_dir)
            evaluator.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Run evaluation
        logger.info("Running comprehensive evaluation...")
        evaluation_results = await evaluator.run_evaluation(
            clear_memory_between=args.clear_memory
        )
        
        # Generate and display report
        report = evaluator.generate_report(evaluation_results)
        
        print("\n" + "=" * 80)
        print("EVALUATION REPORT")
        print("=" * 80)
        print(report)
        print("=" * 80)
        
        # Save report to file
        report_file = evaluator.results_dir / f"evaluation_report_{evaluation_results['timestamp'].replace(':', '-').replace('.', '-')}.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.success(f"📊 Evaluation completed successfully!")
        logger.success(f"📄 Report saved to: {report_file}")
        
        # Print summary statistics
        summary = evaluation_results['summary']
        overall_score = summary['overall_metrics']['overall_score']['mean']
        
        print(f"\n🎯 OVERALL PERFORMANCE SCORE: {overall_score:.3f}/1.000")
        
        if overall_score >= 0.8:
            print("🏆 Excellent performance!")
        elif overall_score >= 0.6:
            print("👍 Good performance!")
        elif overall_score >= 0.4:
            print("⚠️  Needs improvement")
        else:
            print("❌ Poor performance - review system configuration")
        
        return evaluation_results
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {str(e)}")
        logger.exception("Full error details:")
        sys.exit(1)


def run_quick_test():
    """Run a quick test with a single question."""
    
    async def quick_test():
        logger.info("🔧 Running quick test...")
        
        # Initialize RAG system
        rag_system = rag_manager.get_rag_system()
        if not rag_system.is_initialized:
            logger.info("Initializing RAG system...")
            await rag_system.initialize()
        
        # Test with a simple question
        test_question = "What is the transformer architecture?"
        logger.info(f"Test question: {test_question}")
        
        response = await rag_system.query(test_question)
        
        print("\n" + "=" * 50)
        print("QUICK TEST RESULT")
        print("=" * 50)
        print(f"Question: {test_question}")
        print(f"Answer: {response.get('response', 'No response')}")
        print(f"Sources: {len(response.get('sources', []))} found")
        print(f"Top similarity: {response.get('retrieval_stats', {}).get('top_score', 0):.3f}")
        print("=" * 50)
        
        logger.success("✅ Quick test completed!")
    
    asyncio.run(quick_test())


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        run_quick_test()
    else:
        asyncio.run(main()) 