import { useState, useEffect } from 'react';
import { useModelStore } from '../../stores/modelStore';

interface TreeNode {
  name: string;
  fullPath: string;
  moduleType: string | null;
  children: TreeNode[];
}

function buildTree(flat: Record<string, string>): TreeNode {
  const root: TreeNode = { name: 'model', fullPath: '', moduleType: null, children: [] };

  for (const [path, type] of Object.entries(flat)) {
    const parts = path.split('.');
    let current = root;

    for (let i = 0; i < parts.length; i++) {
      const segment = parts[i];
      const fullPath = parts.slice(0, i + 1).join('.');
      let child = current.children.find((c) => c.name === segment);
      if (!child) {
        child = { name: segment, fullPath, moduleType: null, children: [] };
        current.children.push(child);
      }
      if (i === parts.length - 1) {
        child.moduleType = type;
      }
      current = child;
    }
  }

  return root;
}

function getTypeBadge(type: string): { label: string; color: string } {
  const lower = type.toLowerCase();
  if (lower.includes('linear')) return { label: 'Linear', color: 'bg-blue-500/20 text-blue-400' };
  if (lower.includes('attention')) return { label: 'Attn', color: 'bg-purple-500/20 text-purple-400' };
  if (lower.includes('mlp') || lower.includes('feedforward'))
    return { label: 'MLP', color: 'bg-green-500/20 text-green-400' };
  if (lower.includes('norm')) return { label: 'Norm', color: 'bg-yellow-500/20 text-yellow-400' };
  if (lower.includes('embed')) return { label: 'Embed', color: 'bg-cyan-500/20 text-cyan-400' };
  return { label: type.split('.').pop() || type, color: 'bg-zinc-500/20 text-zinc-400' };
}

function TreeNodeItem({ node, depth }: { node: TreeNode; depth: number }) {
  const [expanded, setExpanded] = useState(depth < 1);
  const hasChildren = node.children.length > 0;
  const isLeaf = !hasChildren;

  // Count total leaf modules under this node
  const leafCount = countLeaves(node);

  return (
    <div>
      <div
        className={`flex items-center gap-1.5 rounded px-1.5 py-0.5 text-sm hover:bg-zinc-700/30 ${
          hasChildren ? 'cursor-pointer' : ''
        }`}
        style={{ paddingLeft: `${depth * 16 + 4}px` }}
        onClick={() => hasChildren && setExpanded(!expanded)}
      >
        {/* Chevron */}
        {hasChildren ? (
          <span
            className={`inline-flex h-4 w-4 flex-shrink-0 items-center justify-center text-xs text-zinc-500 transition-transform duration-150 ${
              expanded ? 'rotate-90' : ''
            }`}
          >
            &#9654;
          </span>
        ) : (
          <span className="inline-block h-4 w-4 flex-shrink-0" />
        )}

        {/* Name */}
        <span className={`font-mono text-xs ${isLeaf ? 'text-zinc-300' : 'text-zinc-400'}`}>
          {node.name}
        </span>

        {/* Type badge */}
        {node.moduleType && (
          <span
            className={`rounded px-1.5 py-0.5 text-[10px] font-medium ${getTypeBadge(node.moduleType).color}`}
          >
            {getTypeBadge(node.moduleType).label}
          </span>
        )}

        {/* Child count for collapsed containers */}
        {hasChildren && !expanded && (
          <span className="text-[10px] text-zinc-600">({leafCount})</span>
        )}
      </div>

      {/* Children */}
      {hasChildren && expanded && (
        <div>
          {node.children.map((child) => (
            <TreeNodeItem key={child.fullPath} node={child} depth={depth + 1} />
          ))}
        </div>
      )}
    </div>
  );
}

function countLeaves(node: TreeNode): number {
  if (node.children.length === 0) return 1;
  return node.children.reduce((sum, child) => sum + countLeaves(child), 0);
}

export function ModuleTree() {
  const { moduleTree, loaded, fetchModuleTree } = useModelStore();
  const [treeExpanded, setTreeExpanded] = useState(false);

  useEffect(() => {
    if (loaded && !moduleTree) {
      fetchModuleTree();
    }
  }, [loaded, moduleTree, fetchModuleTree]);

  if (!loaded || !moduleTree) return null;

  const tree = buildTree(moduleTree);
  const totalModules = Object.keys(moduleTree).length;

  return (
    <div className="border-t border-zinc-700/50">
      <button
        onClick={() => setTreeExpanded(!treeExpanded)}
        className="flex w-full items-center justify-between px-5 py-3 text-left transition-colors hover:bg-zinc-800/30"
      >
        <span className="text-xs font-medium uppercase tracking-wider text-zinc-400">
          Module Tree
        </span>
        <span className="flex items-center gap-2">
          <span className="text-xs text-zinc-600">{totalModules} modules</span>
          <span
            className={`text-xs text-zinc-500 transition-transform duration-150 ${
              treeExpanded ? 'rotate-90' : ''
            }`}
          >
            &#9654;
          </span>
        </span>
      </button>

      {treeExpanded && (
        <div className="max-h-96 overflow-y-auto px-3 pb-3">
          {tree.children.map((child) => (
            <TreeNodeItem key={child.fullPath} node={child} depth={0} />
          ))}
        </div>
      )}
    </div>
  );
}
