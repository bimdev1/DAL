"""Tests for pipeline_v3.py"""

import unittest
import time
from unittest.mock import patch, MagicMock, ANY

from dal.pipeline_v3 import run_enhanced_pipeline_v3, PipelineConfig
from dal.local_generator import LocalGenerator
from dal.dal_types import DALTagBlock, DALRunArtifact

def mock_stitch_segments(segments, tags=None):
    """Mock implementation of stitch_segments for testing."""
    return " ".join(segments)

def mock_segment_prompt(prompt, n_segments):
    """Mock implementation of segment_prompt for testing."""
    return [f"Segment {i+1}" for i in range(n_segments)]

def mock_assign_tags(segment):
    """Mock implementation of assign_tags for testing."""
    return ["tag1", "tag2"]

def mock_create_sdt_from_tags(tags):
    """Mock implementation of _create_sdt_from_tags for testing."""
    return DALTagBlock()

class TestPipelineV3(unittest.TestCase):
    """Test cases for pipeline_v3 functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_prompt = "This is a test prompt with multiple sentences. " \
                         "It should be split into segments. Each segment will be processed."
        
        # Mock LocalGenerator
        self.mock_generator = MagicMock(spec=LocalGenerator)
        self.mock_generator.is_ready.return_value = True
        self.mock_generator.generate.return_value = {
            "text": "[GENERATED TEXT]",
            "tokens_in": 10,
            "tokens_out": 20,
            "duration": 0.5,
            "backend": "huggingface",
            "model": "test-model",
            "prompt_preview": "[PROMPT PREVIEW]",
            "sdt_applied": {"tone": True}
        }
        
        # Patch dependencies
        self.patchers = [
            patch('dal.pipeline_v3.LocalGenerator', return_value=self.mock_generator),
            patch('dal.pipeline_v3.segment_prompt', side_effect=mock_segment_prompt),
            patch('dal.pipeline_v3.assign_tags', side_effect=mock_assign_tags),
            patch('dal.pipeline_v3._create_sdt_from_tags', side_effect=mock_create_sdt_from_tags),
            patch('dal.pipeline_v3.stitch_segments', side_effect=mock_stitch_segments)
        ]
        
        for patcher in self.patchers:
            patcher.start()
        
    def tearDown(self):
        """Clean up after tests."""
        for patcher in self.patchers:
            patcher.stop()
    
    def test_pipeline_v3_basic(self):
        """Test basic pipeline_v3 functionality without local generation."""
        # Test with local generation disabled
        config = PipelineConfig(use_local_generation=False)
        result, artifact = run_enhanced_pipeline_v3(
            self.test_prompt, 
            config=config, 
            return_artifact=True
        )
        
        # Verify basic structure
        self.assertIn('text', result)
        self.assertIn('segments', result)
        self.assertIn('tags', result)
        self.assertIn('sdts', result)
        self.assertIn('metrics', result)
        
        # Verify no generation was attempted
        self.mock_generator.generate.assert_not_called()
        
        # Verify artifact is of correct type
        self.assertIsInstance(artifact, DALRunArtifact)
    
    def test_pipeline_v3_with_local_generation(self):
        """Test pipeline_v3 with local generation enabled."""
        # Setup mock return value for generate
        self.mock_generator.generate.return_value = {
            'text': 'Generated text',
            'tokens_in': 10,
            'tokens_out': 20,
            'backend': 'test',
            'model': 'test-model',
            'sdt_applied': True
        }
        
        # Test with local generation enabled
        config = PipelineConfig(
            use_local_generation=True,
            sdt_control=True,
            force_regen=True,  # Ensure we force regeneration
            local_model_spec={
                'backend': 'huggingface',
                'model_name_or_path': 'test-model',
                'load_in_4bit': True
            }
        )
        
        # Create a mock SDT that will trigger regeneration
        mock_sdt = DALTagBlock()
        
        # Patch the SDT creation to return our mock SDT
        with patch('dal.pipeline_v3._create_sdt_from_tags', return_value=mock_sdt):
            result, artifact = run_enhanced_pipeline_v3(
                self.test_prompt,
                config=config,
                return_artifact=True
            )
        
        # Verify generation was attempted for each segment
        self.assertEqual(self.mock_generator.generate.call_count, 3)  # 3 segments
        
        # Check metrics
        metrics = result['metrics']
        self.assertEqual(metrics['generation']['segments_processed'], 3)
        self.assertEqual(metrics['generation']['segments_regenerated'], 3)
        self.assertGreater(metrics['generation']['total_tokens_in'], 0)
        self.assertGreater(metrics['generation']['total_tokens_out'], 0)
        
        # Verify artifact contains expected data
        self.assertEqual(len(artifact.blocks), config.n_segments)
    
    def test_pipeline_v3_force_regen(self):
        """Test pipeline_v3 with force_regeneration=True."""
        # Test with force regeneration
        config = PipelineConfig(
            use_local_generation=True,
            force_regen=True,
            local_model_spec={'backend': 'huggingface'}
        )
        
        result, artifact = run_enhanced_pipeline_v3(
            self.test_prompt,
            config=config,
            return_artifact=True
        )
        
        # Verify generation was attempted for all segments
        self.assertEqual(
            result['metrics']['generation']['segments_processed'],
            len(result['segments'])
        )
        self.assertEqual(
            result['metrics']['generation']['segments_regenerated'],
            len(result['segments'])
        )
        
        # Verify the output contains the mock generated text
        self.assertIn("[GENERATED TEXT]", result['text'])
    
    def test_pipeline_v3_generation_failure(self):
        """Test pipeline_v3 handles generation failures gracefully."""
        # Setup generator to fail
        self.mock_generator.generate.side_effect = Exception("Generation failed")
        
        config = PipelineConfig(
            use_local_generation=True,
            force_regen=True
        )
        
        # The mock segments are 'Segment 1', 'Segment 2', etc.
        # Let's verify the pipeline handles the failure case correctly
        with patch('dal.pipeline_v3._create_sdt_from_tags', return_value=DALTagBlock()):
            result, artifact = run_enhanced_pipeline_v3(
                self.test_prompt,
                config=config,
                return_artifact=True
            )
        
        # The pipeline should still return the original segments when generation fails
        self.assertIn('Segment 1', result['text'])
        self.assertIn('Segment 2', result['text'])
        self.assertIn('Segment 3', result['text'])
        
        # Check metrics structure in failure case
        metrics = result['metrics']
        self.assertIn('generation', metrics)
        self.assertEqual(metrics['generation']['segments_processed'], 3)
        self.assertEqual(metrics['generation']['segments_regenerated'], 0)  # No successful regenerations
        
        # Verify artifact contains the expected number of blocks
        self.assertEqual(len(artifact.blocks), config.n_segments)

if __name__ == '__main__':
    unittest.main()
