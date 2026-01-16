from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from paddleocr import PaddleOCR
from PIL import Image
import tempfile
import os
import numpy as np
from typing import Optional
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="OCR API",
    description="基于PaddleOCR的文字识别服务",
    version="1.0.0"
)

# CORS配置 - 生产环境需要限制具体域名
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境改为具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 配置常量
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
SUPPORTED_LANGS = ['ch', 'en', 'korean', 'japan']

# 初始化OCR模型（支持中英文）
ocr_models = {
    'ch': PaddleOCR(use_angle_cls=True, lang='ch'),
    'en': PaddleOCR(use_angle_cls=True, lang='en')
}

def jsonable(x):
    """递归转换为JSON可序列化对象"""
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [jsonable(v) for v in x]
    return str(x)

def validate_file(file: UploadFile) -> tuple[bool, Optional[str]]:
    """验证上传文件"""
    if not file.filename:
        return False, "文件名为空"
    
    # 检查文件扩展名
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"不支持的文件类型。支持的类型: {', '.join(ALLOWED_EXTENSIONS)}"
    
    return True, None

def save_upload_to_temp(upload: UploadFile, max_size: int = MAX_FILE_SIZE) -> str:
    """保存上传文件到临时目录，带文件大小限制"""
    suffix = os.path.splitext(upload.filename or "")[1].lower() or ".img"
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        # 分块读取，控制大小
        total_size = 0
        while chunk := upload.file.read(8192):
            total_size += len(chunk)
            if total_size > max_size:
                tmp.close()
                os.remove(tmp.name)
                raise HTTPException(
                    status_code=413, 
                    detail=f"文件过大，最大允许 {max_size / 1024 / 1024}MB"
                )
            tmp.write(chunk)
        return tmp.name

@app.get("/")
async def root():
    """健康检查端点"""
    return {
        "status": "running",
        "service": "OCR API",
        "supported_languages": SUPPORTED_LANGS,
        "max_file_size_mb": MAX_FILE_SIZE / 1024 / 1024
    }

@app.get("/health")
async def health_check():
    """详细健康检查"""
    return {
        "status": "healthy",
        "models_loaded": list(ocr_models.keys())
    }

@app.post("/ocr")
async def ocr_image(
    file: UploadFile = File(...),
    lang: str = 'ch',
    return_confidence: bool = True
):
    """
    OCR文字识别接口
    
    - **file**: 图片文件（支持 jpg, png, tiff 等）
    - **lang**: 识别语言（ch=中文, en=英文）
    - **return_confidence**: 是否返回置信度分数
    """
    
    # 验证文件
    is_valid, error_msg = validate_file(file)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)
    
    # 验证语言参数
    if lang not in ocr_models:
        raise HTTPException(
            status_code=400, 
            detail=f"不支持的语言: {lang}。支持: {list(ocr_models.keys())}"
        )
    
    path = None
    tmp_png = None
    
    try:
        # 保存上传文件
        path = save_upload_to_temp(file)
        logger.info(f"处理文件: {file.filename}, 大小: {os.path.getsize(path)} bytes")
        
        ext = os.path.splitext(path)[1].lower()
        
        # TIFF特殊处理
        if ext in [".tif", ".tiff"]:
            img = Image.open(path)
            img.seek(0)
            tmp_png = path + ".png"
            img.convert("RGB").save(tmp_png)
            target_path = tmp_png
        else:
            target_path = path
        
        # 执行OCR识别
        ocr = ocr_models[lang]
        result = ocr.predict(target_path)
        
        pages = []
        for res in result:
            if hasattr(res, "json"):
                data = res.json
            elif hasattr(res, "to_dict"):
                data = res.to_dict()
            else:
                data = res
            pages.append(jsonable(data))
        
        # 提取纯文本
        full_text = ""
        total_confidence = 0
        text_count = 0
        
        for p in pages:
            if isinstance(p, dict):
                texts = p.get("res", {}).get("rec_texts")
                scores = p.get("res", {}).get("rec_scores") if return_confidence else None
                
                if texts:
                    full_text += "\n".join(t for t in texts if t) + "\n"
                    
                if scores:
                    total_confidence += sum(scores)
                    text_count += len(scores)
        
        response = {
            "success": True,
            "filename": file.filename,
            "language": lang,
            "full_text": full_text.strip(),
            "text_count": text_count,
        }
        
        if return_confidence and text_count > 0:
            response["avg_confidence"] = round(total_confidence / text_count, 4)
        
        if return_confidence:
            response["pages"] = pages
        
        logger.info(f"识别完成: {file.filename}, 识别到 {text_count} 个文本块")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"OCR处理失败: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"OCR处理失败: {str(e)}"
        )
    finally:
        # 清理临时文件
        for temp_file in [path, tmp_png]:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception as e:
                    logger.warning(f"清理临时文件失败: {temp_file}, {e}")

@app.post("/ocr/batch")
async def ocr_batch(files: list[UploadFile] = File(...), lang: str = 'ch'):
    """批量OCR识别（最多10个文件）"""
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="批量处理最多10个文件")
    
    results = []
    for file in files:
        try:
            result = await ocr_image(file, lang, return_confidence=False)
            results.append(result)
        except Exception as e:
            results.append({
                "success": False,
                "filename": file.filename,
                "error": str(e)
            })
    
    return {"batch_results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)