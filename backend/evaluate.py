"""Evaluation script for the RAG system."""

import sys
import argparse
from app.rag_system import RAGSystem
from evaluation.evaluation_framework import EvaluationFramework


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate the RAG system")
    parser.add_argument(
        "mode", 
        nargs='?', 
        default="full",
        choices=["full", "quick"],
        help="Evaluation mode: 'full' for all questions, 'quick' for first 3 questions"
    )
    parser.add_argument(
        "--initialize",
        action="store_true",
        help="Initialize the system before evaluation"
    )
    parser.add_argument(
        "--rebuild",
        action="store_true", 
        help="Rebuild the system from scratch before evaluation"
    )
    
    args = parser.parse_args()
    
    print("🔍 RAG System Evaluation")
    print("=" * 40)
    
    # Initialize RAG system
    print("🚀 Initializing RAG system...")
    rag_system = RAGSystem()
    
    try:
        # Initialize system
        if args.rebuild:
            print("🔄 Rebuilding system from scratch...")
            result = rag_system.rebuild_system()
        elif args.initialize or not rag_system.is_initialized:
            print("⚙️  Initializing system...")
            result = rag_system.initialize()
        else:
            # Try to load existing system
            result = rag_system.initialize()
        
        if not result["success"]:
            print(f"❌ System initialization failed: {result.get('error')}")
            sys.exit(1)
        
        print("✅ System ready for evaluation")
        
        # Create evaluation framework
        evaluator = EvaluationFramework(rag_system)
        
        # Run evaluation
        quick_mode = (args.mode == "quick")
        print(f"\n📊 Running {'quick' if quick_mode else 'full'} evaluation...")
        
        results = evaluator.run_evaluation(quick_mode=quick_mode)
        
        # Generate and display report
        print("\n" + "=" * 60)
        print("EVALUATION REPORT")
        print("=" * 60)
        
        report = evaluator.generate_report(results)
        print(report)
        
        # Summary
        if results and "aggregate_statistics" in results:
            stats = results["aggregate_statistics"]
            if "average_overall_score" in stats:
                score = stats["average_overall_score"]
                print(f"\n🎯 FINAL SCORE: {score:.3f}/1.0 ({score*100:.1f}%)")
                
                # Performance categorization
                if score >= 0.8:
                    print("🌟 Excellent performance!")
                elif score >= 0.6:
                    print("👍 Good performance!")
                elif score >= 0.4:
                    print("⚠️  Moderate performance - room for improvement")
                else:
                    print("⚠️  Poor performance - significant improvements needed")
        
        print(f"\n💾 Detailed results saved to: {evaluator.evaluation_results_file}")
        print("✅ Evaluation completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⏹️  Evaluation interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 