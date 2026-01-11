#!/usr/bin/env python3
"""
Test All Mitigation Problems Script

registry에 있는 모든 mitigation problem을 실행하고,
성공한 문제들의 리스트를 출력합니다.
"""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

from aiopslab.orchestrator.orchestrator import Orchestrator
from aiopslab.orchestrator.problems.registry import ProblemRegistry
from clients.registry import AgentRegistry


@dataclass
class MitigationTestResult:
    """단일 mitigation 문제 테스트 결과"""
    problem_id: str
    success: bool
    error: Optional[str]
    results: Optional[Dict]
    execution_time: float
    timestamp: str


class MitigationTester:
    """모든 mitigation 문제를 테스트하는 클래스"""
    
    def __init__(self, agent_name: str = "gpt", max_steps: int = 20, results_dir: str = "experiments/results"):
        self.agent_name = agent_name
        self.max_steps = max_steps
        
        # 실행별 날짜/시간 기반 폴더 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_results_dir = Path(results_dir)
        self.results_dir = base_results_dir / f"mitigation_test_{timestamp}"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Results will be saved to: {self.results_dir}")
        
        self.test_results: List[MitigationTestResult] = []
        
        # Registry 초기화
        self.registry = ProblemRegistry()
        
        # 모든 mitigation 문제 ID 가져오기
        self.mitigation_problem_ids = self.registry.get_problem_ids("mitigation")
        print(f"Found {len(self.mitigation_problem_ids)} mitigation problems:")
        for pid in self.mitigation_problem_ids:
            print(f"  - {pid}")
    
    async def test_single_mitigation(self, problem_id: str) -> MitigationTestResult:
        """단일 mitigation 문제 테스트"""
        print(f"\n{'='*60}")
        print(f"Testing: {problem_id}")
        print(f"{'='*60}")
        
        start_time = time.time()
        error = None
        results = None
        success = False
        
        try:
            # Agent 및 Orchestrator 초기화
            agent_registry = AgentRegistry()
            agent_cls = agent_registry.get_agent(self.agent_name)
            if agent_cls is None:
                raise ValueError(f"Unknown agent: {self.agent_name}")
            
            agent = agent_cls()
            # Path 객체를 그대로 전달 (문자열로 변환하지 않음)
            orchestrator = Orchestrator(results_dir=self.results_dir)
            orchestrator.register_agent(agent, name=f"{self.agent_name}-agent")
            
            # 문제 초기화
            print(f"Initializing problem {problem_id}...")
            problem_desc, instructs, apis = orchestrator.init_problem(problem_id)
            agent.init_context(problem_desc, instructs, apis)
            
            # 문제 실행
            print(f"Running problem {problem_id} (max {self.max_steps} steps)...")
            execution_result = await orchestrator.start_problem(max_steps=self.max_steps)
            
            # 결과 확인
            results = execution_result.get("results", {})
            
            # 성공 여부 판정
            # mitigation 문제의 경우, eval 결과에서 "success" 키 확인
            # 대부분의 mitigation 문제는 self.results["success"] = True/False를 설정함
            if results:
                # "success" 키가 명시적으로 있는 경우 그 값을 사용
                if "success" in results:
                    success = bool(results["success"])
                else:
                    # "success" 키가 없는 경우, TTM이 있고 값이 유효하면 성공으로 간주
                    # (submit()이 호출되어 정상적으로 완료된 경우)
                    success = "TTM" in results and results.get("TTM") is not None
            else:
                # results가 없으면 실패로 간주
                success = False
            
            if success:
                print(f"✓ SUCCESS: {problem_id}")
            else:
                print(f"✗ FAILED: {problem_id}")
                if results:
                    print(f"  Results: {results}")
            
        except Exception as e:
            error = str(e)
            error_str = str(e).lower()
            
            # Rate limit 에러인 경우 특별 처리
            is_rate_limit = "rate limit" in error_str or "429" in error_str or "rate_limit" in error_str
            
            print(f"✗ ERROR: {problem_id}")
            if is_rate_limit:
                print(f"  Rate Limit Error: {error}")
                print(f"  Waiting 60 seconds before continuing to next problem...")
                await asyncio.sleep(60)  # Rate limit 발생 시 더 긴 대기
            else:
                print(f"  Error: {error}")
                import traceback
                traceback.print_exc()
        
        execution_time = time.time() - start_time
        
        result = MitigationTestResult(
            problem_id=problem_id,
            success=success,
            error=error,
            results=results,
            execution_time=execution_time,
            timestamp=datetime.now().isoformat()
        )
        
        return result
    
    async def test_all_mitigations(self):
        """모든 mitigation 문제 테스트"""
        print(f"\n{'#'*60}")
        print(f"Testing All Mitigation Problems")
        print(f"Agent: {self.agent_name}")
        print(f"Max Steps: {self.max_steps}")
        print(f"Total Problems: {len(self.mitigation_problem_ids)}")
        print(f"{'#'*60}\n")
        
        for i, problem_id in enumerate(self.mitigation_problem_ids, 1):
            print(f"\n[{i}/{len(self.mitigation_problem_ids)}]")
            result = await self.test_single_mitigation(problem_id)
            self.test_results.append(result)
            
            # 문제 간 대기 (리소스 정리 시간)
            # Rate limit 에러가 발생한 경우 더 긴 대기
            if i < len(self.mitigation_problem_ids):
                wait_time = 30 if result.error and ("rate limit" in result.error.lower() or "429" in result.error) else 10
                print(f"\nWaiting {wait_time} seconds before next test...")
                await asyncio.sleep(wait_time)
        
        # 결과 저장 및 출력
        self.save_results()
        self.print_summary()
    
    def save_results(self):
        """결과 저장"""
        # 이미 실행별 폴더가 있으므로 파일명만 사용
        results_file = self.results_dir / "mitigation_test_results.json"
        
        results_data = {
            "agent": self.agent_name,
            "max_steps": self.max_steps,
            "total_problems": len(self.mitigation_problem_ids),
            "test_results": [asdict(r) for r in self.test_results]
        }
        
        with open(results_file, "w") as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")
    
    def print_summary(self):
        """결과 요약 출력"""
        print(f"\n{'='*60}")
        print("Test Summary")
        print(f"{'='*60}\n")
        
        total = len(self.test_results)
        successful = [r for r in self.test_results if r.success]
        failed = [r for r in self.test_results if not r.success]
        
        print(f"Total Problems Tested: {total}")
        print(f"Successful: {len(successful)} ({len(successful)/total*100:.1f}%)")
        print(f"Failed: {len(failed)} ({len(failed)/total*100:.1f}%)")
        
        if successful:
            print(f"\n{'='*60}")
            print("SUCCESSFUL Mitigation Problems:")
            print(f"{'='*60}")
            for result in successful:
                print(f"  ✓ {result.problem_id}")
                if result.results:
                    # 주요 결과만 출력
                    if "TTM" in result.results:
                        print(f"    TTM: {result.results['TTM']}")
                print(f"    Execution Time: {result.execution_time:.2f}s")
            
            # 성공한 문제 ID만 리스트로 출력
            print(f"\n{'='*60}")
            print("SUCCESSFUL Problem IDs (for copy-paste):")
            print(f"{'='*60}")
            successful_ids = [r.problem_id for r in successful]
            for pid in successful_ids:
                print(f"  {pid}")
            
            # JSON 형식으로도 출력
            print(f"\n{'='*60}")
            print("SUCCESSFUL Problem IDs (JSON):")
            print(f"{'='*60}")
            print(json.dumps(successful_ids, indent=2))
        
        if failed:
            print(f"\n{'='*60}")
            print("FAILED Mitigation Problems:")
            print(f"{'='*60}")
            for result in failed:
                print(f"  ✗ {result.problem_id}")
                if result.error:
                    print(f"    Error: {result.error}")
                if result.results:
                    print(f"    Results: {result.results}")
        
        print(f"\n{'='*60}")


async def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test all mitigation problems in the registry",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 기본 실행 (gpt agent, 20 steps)
  python3 test_all_mitigations.py
  
  # 다른 agent 사용
  python3 test_all_mitigations.py --agent qwen
  
  # 최대 스텝 수 변경
  python3 test_all_mitigations.py --max-steps 30
        """
    )
    parser.add_argument("--agent", type=str, default="gpt",
                       choices=["gpt", "qwen", "deepseek", "vllm"],
                       help="Agent name (default: gpt)")
    parser.add_argument("--max-steps", type=int, default=20,
                       help="Max steps for agent (default: 20)")
    parser.add_argument("--results-dir", type=str, default="experiments/results",
                       help="Results directory (default: experiments/results)")
    
    args = parser.parse_args()
    
    tester = MitigationTester(
        agent_name=args.agent,
        max_steps=args.max_steps,
        results_dir=args.results_dir
    )
    
    await tester.test_all_mitigations()


if __name__ == "__main__":
    asyncio.run(main())
