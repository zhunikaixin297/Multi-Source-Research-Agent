import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

from docling.document_converter import DocumentConverter
from docling_core.types.doc import DescriptionAnnotation

from ...domain.interfaces import DocumentParser
from ...domain.models import DocumentSource
from ..repository.session_memory_manager import SessionMemoryManager, get_session_memory_manager

log = logging.getLogger(__name__)

class DoclingParser(DocumentParser):
    IMAGE_PLACEHOLDER_PATTERN = re.compile(r"!\[[^\]]*\]\(([^)]+)\)|<!--\s*image\s*-->")

    def __init__(
        self,
        converter: DocumentConverter,
        max_concurrent_docs: int = 1,
        session_manager: Optional[SessionMemoryManager] = None,
    ):
        """
        初始化 DoclingParser。
        
        所有复杂的配置和实例化逻辑都已移至工厂方法中。
        此处仅接收已构建好的 converter 实例。

        :param converter: 已配置好 VLM Pipeline 的 DocumentConverter 实例
        :param max_concurrent_docs: 全局文档处理并发限制
        """
        log.info("DoclingParser 初始化 (依赖注入)...")
        self.doc_converter = converter
        self.parse_semaphore = asyncio.Semaphore(max_concurrent_docs)
        self.session_manager = session_manager or get_session_memory_manager()
        log.info(f"DoclingParser 就绪。并发限制: {max_concurrent_docs}")

    @staticmethod
    def _extract_picture_caption(picture: Any) -> Optional[str]:
        annotations = getattr(picture, "annotations", []) or []
        for annotation in annotations:
            if isinstance(annotation, DescriptionAnnotation) and annotation.text:
                return annotation.text.strip()
        return None

    def _export_workspace_images(
        self,
        doc: Any,
        workspace_id: str,
        document_id: str,
    ) -> List[Dict[str, Any]]:
        images_dir = self.session_manager.init_workspace_env(workspace_id)
        images_metadata: List[Dict[str, Any]] = []
        pictures = getattr(doc, "pictures", []) or []
        for idx, picture in enumerate(pictures, start=1):
            get_image = getattr(picture, "get_image", None)
            if not callable(get_image):
                continue
            pil_image = get_image(doc)
            if pil_image is None:
                continue
            image_id = f"{document_id[:8]}_{idx}"
            image_path = images_dir / f"{image_id}.png"
            pil_image.save(image_path, format="PNG")
            caption = self._extract_picture_caption(picture)
            images_metadata.append(
                {
                    "id": image_id,
                    "path": str(image_path),
                    "caption": caption,
                }
            )
        return images_metadata

    @staticmethod
    def _caption_suffix(caption: Optional[str]) -> str:
        if not caption:
            return ""
        return f"\n(Description: {caption})"

    def _inject_image_placeholders_with_caption(
        self,
        markdown: str,
        images_metadata: List[Dict[str, Any]],
    ) -> str:
        matches = list(self.IMAGE_PLACEHOLDER_PATTERN.finditer(markdown))
        if not matches:
            return markdown
        output = markdown
        offset = 0
        for idx, match in enumerate(matches, start=1):
            image = images_metadata[idx - 1] if idx <= len(images_metadata) else None
            image_id = image.get("id", f"image_{idx}") if image else f"image_{idx}"
            caption = image.get("caption") if image else None
            replacement = f"[IMAGE: {image_id}]{self._caption_suffix(caption)}"
            start, end = match.span()
            start += offset
            end += offset
            output = output[:start] + replacement + output[end:]
            offset += len(replacement) - (end - start)
        return output

    def _append_missing_placeholders(
        self,
        markdown: str,
        images_metadata: List[Dict[str, Any]],
    ) -> str:
        if not images_metadata:
            return markdown
        missing = []
        for image in images_metadata:
            tag = f"[IMAGE: {image['id']}]"
            if tag not in markdown:
                missing.append(f"{tag}{self._caption_suffix(image.get('caption'))}")
        if not missing:
            return markdown
        tail = "\n".join(missing)
        return f"{markdown.rstrip()}\n\n{tail}\n"

    async def parse(self, source: DocumentSource) -> str:
        """
        异步解析原始文档，使用 Semaphore 控制并发。
        """
        async with self.parse_semaphore:
            input_doc_path = source.file_path
            
            if not input_doc_path.exists():
                log.error(f"文件未找到: {input_doc_path}")
                raise FileNotFoundError(f"文件未找到: {input_doc_path}")
            
            try:
                def _blocking_convert_and_export():
                    res = self.doc_converter.convert(input_doc_path)
                    doc = res.document
                    md_content = doc.export_to_markdown()
                    source.metadata = source.metadata or {}
                    workspace_id = source.metadata.get("workspace_id")
                    if not workspace_id:
                        return md_content, []
                    images = self._export_workspace_images(
                        doc=doc,
                        workspace_id=workspace_id,
                        document_id=source.document_id,
                    )
                    md_with_images = self._inject_image_placeholders_with_caption(
                        markdown=md_content,
                        images_metadata=images,
                    )
                    md_with_images = self._append_missing_placeholders(md_with_images, images)
                    return md_with_images, images

                md_content, images = await asyncio.to_thread(_blocking_convert_and_export)
                source.metadata = source.metadata or {}
                if images:
                    source.metadata["images"] = images
                log.info(f"转换完成: {source.file_path.name}")
                return md_content

            except Exception as e:
                log.error(f"解析文档时出错 {input_doc_path.name}: {e}", exc_info=True)
                raise
