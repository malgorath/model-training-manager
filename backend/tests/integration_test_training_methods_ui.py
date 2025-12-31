"""
Integration tests for all training methods via UI simulation.

Following TDD methodology: These tests simulate a real user using the frontend UI,
using the actual database so results are visible in the dashboard.

Tests all training methods:
- QLoRA
- Unsloth
- RAG
- Standard

Each test:
1. Creates a small dataset (20 rows)
2. Loads available model from dropdown
3. Creates a project via API
4. Starts training
5. Waits for completion
6. Verifies results are visible in dashboard
"""

import os
import sys
import time
import json
import requests
from pathlib import Path
from typing import Dict, Any, Optional

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

BASE_URL = "http://localhost:8000"
API_BASE = f"{BASE_URL}/api/v1"


class UITrainingTester:
    """
    Simulates a user using the frontend UI to test training methods.
    
    Uses actual database so results are visible in dashboard.
    """
    
    def __init__(self):
        self.session = requests.Session()
        self.datasets: Dict[str, int] = {}
        self.projects: Dict[str, int] = {}
    
    def get_existing_dataset(self, min_rows: int = 20) -> Optional[int]:
        """
        Get an existing dataset from the database.
        
        Args:
            min_rows: Minimum number of rows required
            
        Returns:
            Dataset ID or None if not found
        """
        print(f"\n📊 Finding existing dataset with at least {min_rows} rows...")
        
        response = self.session.get(f"{API_BASE}/datasets/?page=1&page_size=100")
        response.raise_for_status()
        datasets_data = response.json()
        
        # Find a suitable dataset (prefer smaller ones for faster testing)
        suitable_datasets = [
            d for d in datasets_data['items']
            if d.get('row_count', 0) >= min_rows
        ]
        
        if not suitable_datasets:
            print(f"❌ No datasets found with at least {min_rows} rows")
            return None
        
        # Prefer smaller datasets for faster testing
        suitable_datasets.sort(key=lambda x: x.get('row_count', 0))
        dataset = suitable_datasets[0]
        dataset_id = dataset['id']
        
        print(f"✅ Using existing dataset: ID {dataset_id} - {dataset['name']} ({dataset.get('row_count', 0)} rows)")
        return dataset_id
    
    def get_available_models(self) -> list[str]:
        """
        Get available models from dropdown (simulating user selecting model).
        
        Returns:
            List of available model IDs
        """
        print("\n🔍 Fetching available models from dropdown...")
        response = self.session.get(f"{API_BASE}/projects/models/available")
        response.raise_for_status()
        models = response.json()
        print(f"✅ Found {len(models)} available models: {models}")
        return models
    
    def get_model_types(self, model_name: str) -> Optional[str]:
        """
        Get model types for a model and return the recommended model_type.
        
        Args:
            model_name: Model identifier
            
        Returns:
            Recommended model_type or None if not available
        """
        try:
            response = self.session.get(f"{API_BASE}/projects/models/{model_name}/types")
            response.raise_for_status()
            data = response.json()
            return data.get('recommended') or data.get('model_type')
        except Exception as e:
            print(f"⚠️  Warning: Could not get model types for {model_name}: {e}")
            return None
    
    def create_project(
        self,
        name: str,
        base_model: str,
        training_type: str,
        dataset_id: int,
    ) -> int:
        """
        Create a project via API (simulating user creating project in UI).
        
        Args:
            name: Project name
            base_model: Model to use
            training_type: Training method (qlora, unsloth, rag, standard)
            dataset_id: Dataset ID to use
            
        Returns:
            Project ID
        """
        print(f"\n📝 Creating project: {name} ({training_type})...")
        
        # Fetch model_type for the selected model
        model_type = self.get_model_types(base_model)
        if model_type:
            print(f"   Auto-detected model_type: {model_type}")
        
        project_data = {
            "name": name,
            "description": f"Integration test for {training_type} training method",
            "base_model": base_model,
            "training_type": training_type,
            "output_directory": f"./output/test-{training_type}",
            "traits": [
                {
                    "trait_type": "reasoning",
                    "datasets": [
                        {
                            "dataset_id": dataset_id,
                            "percentage": 100.0
                        }
                    ]
                }
            ]
        }
        
        # Add model_type if detected
        if model_type:
            project_data["model_type"] = model_type
        
        response = self.session.post(
            f"{API_BASE}/projects",
            json=project_data
        )
        if response.status_code != 201:
            error_detail = response.text
            print(f"❌ Project creation failed: {response.status_code}")
            print(f"   Error: {error_detail}")
            try:
                error_json = response.json()
                print(f"   Detail: {error_json.get('detail', 'Unknown error')}")
            except:
                pass
            response.raise_for_status()
        project = response.json()
        project_id = project['id']
        self.projects[name] = project_id
        print(f"✅ Project created: ID {project_id}")
        return project_id
    
    def start_training(self, project_id: int) -> Dict[str, Any]:
        """
        Start training for a project (simulating user clicking "Start Training").
        
        Args:
            project_id: Project ID
            
        Returns:
            Project status
        """
        print(f"\n🚀 Starting training for project {project_id}...")
        
        response = self.session.post(f"{API_BASE}/projects/{project_id}/start")
        response.raise_for_status()
        project = response.json()
        print(f"✅ Training started: Status = {project['status']}")
        return project
    
    def wait_for_completion(
        self,
        project_id: int,
        timeout: int = 300,
        poll_interval: int = 5
    ) -> Dict[str, Any]:
        """
        Wait for training to complete (simulating user watching dashboard).
        
        Args:
            project_id: Project ID
            timeout: Maximum time to wait (seconds)
            poll_interval: Time between status checks (seconds)
            
        Returns:
            Final project status
        """
        print(f"\n⏳ Waiting for training to complete (timeout: {timeout}s = {timeout//60} minutes)...")
        start_time = time.time()
        last_progress = -1
        last_status = None
        
        while time.time() - start_time < timeout:
            try:
                response = self.session.get(f"{API_BASE}/projects/{project_id}", timeout=10)
                response.raise_for_status()
                project = response.json()
                
                status = project['status']
                progress = project.get('progress', 0)
                current_epoch = project.get('current_epoch', 0)
                total_epochs = project.get('epochs', 0) or 3  # Default if not set
                
                # Only print if status or progress changed
                if status != last_status or abs(progress - last_progress) >= 1.0:
                    elapsed = int(time.time() - start_time)
                    print(f"  [{elapsed:4d}s] Status: {status:10s} | Progress: {progress:5.1f}% | Epoch: {current_epoch}/{total_epochs}")
                    last_status = status
                    last_progress = progress
                
                if status in ['completed', 'failed', 'cancelled']:
                    elapsed = int(time.time() - start_time)
                    print(f"\n✅ Training finished after {elapsed}s: Status = {status}")
                    return project
                
                time.sleep(poll_interval)
            except requests.exceptions.RequestException as e:
                print(f"\n⚠️  Error checking status: {e}, retrying...")
                time.sleep(poll_interval)
        
        # Get final status even if timeout
        try:
            response = self.session.get(f"{API_BASE}/projects/{project_id}")
            response.raise_for_status()
            project = response.json()
            elapsed = int(time.time() - start_time)
            print(f"\n⏱️  Timeout reached after {elapsed}s. Final status: {project['status']}")
            return project
        except Exception as e:
            print(f"\n⚠️  Could not get final status: {e}")
            return {"status": "timeout", "progress": 0, "error_message": f"Timeout after {timeout}s"}
    
    def verify_dashboard_visibility(self, project_id: int) -> bool:
        """
        Verify project is visible in dashboard.
        
        Args:
            project_id: Project ID
            
        Returns:
            True if visible
        """
        print(f"\n🔍 Verifying project {project_id} is visible in dashboard...")
        
        # Check projects list
        response = self.session.get(f"{API_BASE}/projects?page=1&page_size=100")
        response.raise_for_status()
        projects_data = response.json()
        project_ids = [p['id'] for p in projects_data['items']]
        
        # Check training jobs list (projects should appear there too)
        response = self.session.get(f"{API_BASE}/jobs?page=1&page_size=100")
        response.raise_for_status()
        jobs_data = response.json()
        job_ids = [j['id'] for j in jobs_data['items']]
        
        visible = project_id in project_ids or project_id in job_ids
        if visible:
            print(f"✅ Project {project_id} is visible in dashboard")
        else:
            print(f"❌ Project {project_id} is NOT visible in dashboard")
        
        return visible
    
    def test_training_method(
        self,
        training_type: str,
        model: str,
        dataset_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Test a specific training method end-to-end.
        
        Args:
            training_type: Training method (qlora, unsloth, rag, standard)
            model: Model to use
            dataset_id: Dataset ID to use (if None, finds existing dataset)
            
        Returns:
            Test results
        """
        print(f"\n{'='*60}")
        print(f"🧪 Testing {training_type.upper()} Training Method")
        print(f"{'='*60}")
        
        try:
            # Step 1: Get existing dataset (or use provided one)
            if dataset_id is None:
                dataset_id = self.get_existing_dataset(min_rows=20)
                if dataset_id is None:
                    raise ValueError("No suitable existing dataset found")
            
            # Step 2: Create project
            project_name = f"test-{training_type}-training"
            project_id = self.create_project(
                name=project_name,
                base_model=model,
                training_type=training_type,
                dataset_id=dataset_id,
            )
            
            # Step 3: Start training
            self.start_training(project_id)
            
            # Step 4: Wait for completion (10 minutes per method)
            print(f"\n⏳ Waiting for {training_type} training to complete...")
            final_status = self.wait_for_completion(project_id, timeout=600)
            
            # Verify training actually ran
            if final_status['status'] == 'pending':
                print(f"⚠️  Warning: Training is still pending after timeout")
            elif final_status['status'] == 'failed':
                print(f"❌ Training failed: {final_status.get('error_message', 'Unknown error')}")
            elif final_status['status'] == 'completed':
                print(f"✅ Training completed successfully!")
                print(f"   Final progress: {final_status.get('progress', 0):.1f}%")
                print(f"   Epochs completed: {final_status.get('current_epoch', 0)}")
                if final_status.get('model_path'):
                    print(f"   Model saved to: {final_status['model_path']}")
            
            # Step 5: Verify dashboard visibility
            visible = self.verify_dashboard_visibility(project_id)
            
            success = final_status['status'] == 'completed' and visible
            
            result = {
                "training_type": training_type,
                "project_id": project_id,
                "dataset_id": dataset_id,
                "status": final_status['status'],
                "progress": final_status.get('progress', 0),
                "current_epoch": final_status.get('current_epoch', 0),
                "total_epochs": final_status.get('epochs', 0),
                "model_path": final_status.get('model_path'),
                "visible_in_dashboard": visible,
                "success": success,
                "error_message": final_status.get('error_message'),
            }
            
            if success:
                print(f"\n✅ {training_type.upper()} test PASSED")
            else:
                print(f"\n❌ {training_type.upper()} test FAILED")
                if final_status['status'] != 'completed':
                    print(f"   Reason: Training status is '{final_status['status']}', not 'completed'")
                if not visible:
                    print(f"   Reason: Project not visible in dashboard")
            
            return result
            
        except Exception as e:
            print(f"\n❌ Test failed: {str(e)}")
            return {
                "training_type": training_type,
                "project_id": project_id if 'project_id' in locals() else None,
                "status": "error",
                "success": False,
                "error_message": str(e),
            }


def clean_projects_and_jobs():
    """
    Clean all projects and training jobs from database.
    Preserves datasets, models, and settings.
    """
    print("\n🧹 Cleaning projects and jobs from database...")
    try:
        from app.core.database import SessionLocal
        from app.models.project import Project, ProjectTrait, ProjectTraitDataset
        from app.models.training_job import TrainingJob
        
        db = SessionLocal()
        try:
            # Delete in correct order due to foreign key constraints
            num_project_trait_datasets = db.query(ProjectTraitDataset).count()
            if num_project_trait_datasets > 0:
                db.query(ProjectTraitDataset).delete()
                print(f"   Deleted {num_project_trait_datasets} project trait datasets")
            
            num_project_traits = db.query(ProjectTrait).count()
            if num_project_traits > 0:
                db.query(ProjectTrait).delete()
                print(f"   Deleted {num_project_traits} project traits")
            
            num_projects = db.query(Project).count()
            if num_projects > 0:
                db.query(Project).delete()
                print(f"   Deleted {num_projects} projects")
            
            num_training_jobs = db.query(TrainingJob).count()
            if num_training_jobs > 0:
                db.query(TrainingJob).delete()
                print(f"   Deleted {num_training_jobs} training jobs")
            
            db.commit()
            print("✅ Cleanup complete!")
        except Exception as e:
            db.rollback()
            print(f"❌ Error during cleanup: {e}")
            raise
        finally:
            db.close()
    except Exception as e:
        print(f"⚠️  Warning: Could not clean database: {e}")


def main():
    """
    Run integration tests for all training methods.
    
    Simulates a real user using the frontend UI to test all training methods.
    Results will be visible in the dashboard.
    
    Clears all projects and jobs before running to ensure clean state.
    """
    # Clean projects and jobs before running tests
    clean_projects_and_jobs()
    print("\n" + "="*60)
    print("🎯 UI Integration Tests for All Training Methods")
    print("="*60)
    print("\nThis will:")
    print("1. Create test datasets (20 rows each)")
    print("2. Load available models from dropdown")
    print("3. Create projects for each training method")
    print("4. Start training and wait for completion")
    print("5. Verify results are visible in dashboard")
    print("\nResults will be visible in the dashboard for review.")
    print("="*60)
    
    tester = UITrainingTester()
    
    # Get available models
    models = tester.get_available_models()
    if not models:
        print("\n❌ No models available. Please download a model first.")
        return
    
    # Use first available model
    test_model = models[0]
    print(f"\n✅ Using model: {test_model}")
    
    # Test ALL training methods (must match TrainingType enum)
    # From backend/app/models/training_job.py: QLORA, UNSLOTH, RAG, STANDARD
    training_methods = ['qlora', 'unsloth', 'rag', 'standard']
    results = []
    
    print(f"\n📋 Will test {len(training_methods)} training methods:")
    for i, method in enumerate(training_methods, 1):
        print(f"   {i}. {method.upper()}")
    
    for method in training_methods:
        print(f"\n{'='*60}")
        print(f"Starting test for {method.upper()}...")
        print(f"{'='*60}")
        
        # Get a dataset to use for all tests (reuse same dataset)
        if method == training_methods[0]:
            # Get dataset on first iteration
            dataset_id = tester.get_existing_dataset(min_rows=20)
            if dataset_id is None:
                print("\n❌ No suitable existing dataset found. Please ensure datasets exist in database.")
                return
        
        result = tester.test_training_method(
            training_type=method,
            model=test_model,
            dataset_id=dataset_id
        )
        results.append(result)
        
        # Wait a bit between tests to avoid overwhelming the system
        if method != training_methods[-1]:  # Don't wait after last test
            print(f"\n⏸️  Waiting 5 seconds before next test...")
            time.sleep(5)
    
    # Print summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for r in results if r['success'])
    total = len(results)
    
    print(f"\n{'='*60}")
    print(f"📊 FINAL RESULTS: {passed}/{total} tests passed")
    print(f"{'='*60}")
    
    for result in results:
        status_icon = "✅" if result['success'] else "❌"
        print(f"\n{status_icon} {result['training_type'].upper()}:")
        print(f"   Project ID: {result.get('project_id', 'N/A')}")
        print(f"   Dataset ID: {result.get('dataset_id', 'N/A')}")
        print(f"   Status: {result['status']}")
        print(f"   Progress: {result.get('progress', 0):.1f}%")
        if result.get('current_epoch') and result.get('total_epochs'):
            print(f"   Epochs: {result['current_epoch']}/{result['total_epochs']}")
        if result.get('model_path'):
            print(f"   Model Path: {result['model_path']}")
        print(f"   Visible in Dashboard: {'✅' if result.get('visible_in_dashboard') else '❌'}")
        if result.get('error_message'):
            print(f"   Error: {result['error_message']}")
    
    print(f"\n{'='*60}")
    if passed == total:
        print("✅ ALL TESTS PASSED! All training methods work correctly.")
    else:
        print(f"⚠️  {total - passed} test(s) failed. Check dashboard for details.")
    print(f"{'='*60}")
    print(f"\n📋 Review all projects in the dashboard:")
    print(f"   - Projects page: http://localhost:3001/projects")
    print(f"   - Training Jobs page: http://localhost:3001/training")
    print(f"   - Dashboard: http://localhost:3001/dashboard")
    print(f"\nProject IDs created:")
    for result in results:
        if result.get('project_id'):
            print(f"   - {result['training_type'].upper()}: Project #{result['project_id']}")
    
    # Return results for programmatic access
    return results


if __name__ == "__main__":
    main()
