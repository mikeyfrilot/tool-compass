# Tool-Compass Test Coverage Requirements

**Goal:** Reach 100% coverage across gateway, indexing, backend management, and semantic routing.

**Current State:**
- Source modules: *.py files in root
- Tests exist but comprehensive coverage needed

---

## 1) gateway.py
**Priority: CRITICAL**
- `test_gateway_initialization`
- `test_gateway_compass_search`
- `test_gateway_describe_tool`
- `test_gateway_execute_tool`
- `test_gateway_list_tools`
- `test_gateway_get_stats`
- `test_gateway_backend_routing`
- `test_gateway_error_handling`
- `test_gateway_concurrent_requests`
- `test_gateway_progressive_disclosure`

## 2) indexer.py (CompassIndex)
**Priority: CRITICAL**
- `test_index_initialization`
- `test_index_build_from_tools`
- `test_index_semantic_search`
- `test_index_save_load`
- `test_index_incremental_update`
- `test_index_empty_index`
- `test_index_large_tool_set`
- `test_index_search_ranking`
- `test_index_hnsw_parameters`
- `test_index_embedding_caching`

## 3) chain_indexer.py
**Priority: HIGH**
- `test_chain_indexer_multi_backend_search`
- `test_chain_indexer_result_merging`
- `test_chain_indexer_duplicate_removal`
- `test_chain_indexer_ranking`
- `test_chain_indexer_empty_results`
- `test_chain_indexer_single_backend`

## 4) embedder.py
**Priority: HIGH**
- `test_embedder_initialization`
- `test_embedder_embed_text`
- `test_embedder_batch_embedding`
- `test_embedder_caching`
- `test_embedder_ollama_connection`
- `test_embedder_timeout`
- `test_embedder_error_handling`
- `test_embedder_fallback`

## 5) config.py
**Priority: CRITICAL**
- `test_config_load_from_file`
- `test_config_load_from_env`
- `test_config_backend_parsing`
- `test_config_validation`
- `test_config_missing_file`
- `test_config_invalid_json`
- `test_config_stdio_backend`
- `test_config_merge_env_overrides`

## 6) backend_client.py
**Priority: CRITICAL**
- `test_backend_connect`
- `test_backend_disconnect`
- `test_backend_list_tools`
- `test_backend_call_tool`
- `test_backend_stdio_subprocess`
- `test_backend_timeout`
- `test_backend_error_handling`
- `test_backend_concurrent_calls`

## 7) backend_client_simple.py
**Priority: HIGH**
- `test_simple_backend_manager`
- `test_simple_backend_connect`
- `test_simple_backend_disconnect`
- `test_simple_backend_list_tools`
- `test_simple_backend_call_tool`
- `test_simple_backend_no_anyio_conflicts`

## 8) tool_manifest.py
**Priority: HIGH**
- `test_tool_definition_parsing`
- `test_tool_definition_validation`
- `test_tool_definition_serialization`
- `test_tool_manifest_from_mcp_schema`
- `test_tool_manifest_invalid_data`

## 9) sync_manager.py
**Priority: HIGH**
- `test_sync_manager_sync_backends`
- `test_sync_manager_rebuild_index`
- `test_sync_manager_incremental_sync`
- `test_sync_manager_error_recovery`
- `test_sync_manager_sync_status`
- `test_sync_manager_concurrent_sync`

## 10) analytics.py
**Priority: MEDIUM**
- `test_analytics_track_search`
- `test_analytics_track_execution`
- `test_analytics_track_error`
- `test_analytics_get_stats`
- `test_analytics_reset`
- `test_analytics_persistence`

## 11) ui.py
**Priority: LOW**
- `test_ui_render_results`
- `test_ui_format_tool_description`
- `test_ui_color_coding`
- `test_ui_pagination`

## 12) Integration Tests
**Priority: CRITICAL**
- `test_end_to_end_tool_discovery`
- `test_end_to_end_tool_execution`
- `test_progressive_disclosure_flow`
- `test_multi_backend_routing`
- `test_index_rebuild_workflow`
- `test_concurrent_tool_calls`
- `test_backend_failure_recovery`
- `test_search_relevance_ranking`

---

## Suggested Test Layout
```
discovery/tool-compass/tests/
  test_gateway.py
  test_indexer.py
  test_chain_indexer.py
  test_embedder.py
  test_config.py
  test_backend_client.py
  test_backend_client_simple.py
  test_tool_manifest.py
  test_sync_manager.py
  test_analytics.py
  test_ui.py
  test_integration.py
  fixtures/
    sample_config.json
    sample_tools.json
```

---

## Notes
- Mock MCP server responses for backend tests
- Mock Ollama embedder for speed
- Test with various backend configurations
- Include semantic search relevance tests
- Test progressive disclosure pattern thoroughly
- Validate token reduction metrics
- Test concurrent backend operations
- Include error recovery scenarios
- Test index persistence and loading
- Validate HNSW search accuracy
