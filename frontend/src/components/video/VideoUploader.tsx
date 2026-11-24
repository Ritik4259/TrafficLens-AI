import React, { useState, useRef, DragEvent, ChangeEvent } from "react";
import {
  UploadCloud,
  FileVideo,
  AlertTriangle,
  CheckCircle2,
  X,
  Play,
  RotateCw,
} from "lucide-react";
import { Button } from "../common/Button";
import { Badge } from "../common/Badge";
import { videoService } from "../../services/videoService";
import { VideoMetadata } from "../../types/video";
import {
  validateVideoFile,
  formatBytes,
  formatDuration,
  ALLOWED_VIDEO_EXTENSIONS,
} from "../../core/validation";
import { APP_CONFIG } from "../../core/config";

interface VideoUploaderProps {
  onUploadSuccess?: (metadata: VideoMetadata) => void;
  onClose?: () => void;
}

export const VideoUploader: React.FC<VideoUploaderProps> = ({
  onUploadSuccess,
  onClose,
}) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [validationError, setValidationError] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [uploadedVideo, setUploadedVideo] = useState<VideoMetadata | null>(null);
  const [isLoadingSample, setIsLoadingSample] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelection = (file: File) => {
    setUploadError(null);
    setUploadedVideo(null);
    setUploadProgress(0);

    const validation = validateVideoFile(file);
    if (!validation.valid) {
      setSelectedFile(null);
      setValidationError(validation.error || "Invalid video file.");
      return;
    }

    setValidationError(null);
    setSelectedFile(file);
  };

  const onFileInputChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFileSelection(e.target.files[0]);
    }
  };

  const onDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const onDragLeave = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const onDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileSelection(e.dataTransfer.files[0]);
    }
  };

  const handleLoadSample = async () => {
    if (!APP_CONFIG.isDev) return;
    setIsLoadingSample(true);
    setUploadError(null);
    setValidationError(null);
    try {
      const response = await fetch("/dev-samples/MVI_20011_sample.mp4");
      if (!response.ok) {
        throw new Error(`Could not load development sample fixture (${response.status} ${response.statusText})`);
      }
      const blob = await response.blob();
      const sampleFile = new File([blob], "MVI_20011_sample.mp4", {
        type: "video/mp4",
      });
      handleFileSelection(sampleFile);
    } catch (err: any) {
      setValidationError(`Failed to load benchmark fixture: ${err.message}`);
    } finally {
      setIsLoadingSample(false);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile || isUploading) return;

    setIsUploading(true);
    setUploadProgress(0);
    setUploadError(null);

    try {
      const response = await videoService.uploadVideo(selectedFile, (percent) => {
        setUploadProgress(percent);
      });

      setUploadedVideo(response.metadata);
      if (onUploadSuccess) {
        onUploadSuccess(response.metadata);
      }
    } catch (err: any) {
      const message =
        err.detail || err.message || "Failed to upload and validate video.";
      setUploadError(typeof message === "string" ? message : JSON.stringify(message));
    } finally {
      setIsUploading(false);
    }
  };

  const resetForm = () => {
    setSelectedFile(null);
    setValidationError(null);
    setUploadError(null);
    setUploadProgress(0);
    setUploadedVideo(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  return (
    <div className="bg-[#111827] border border-[#1F2937] rounded-xl p-6 shadow-2xl max-w-xl w-full text-[#E5E7EB] font-sans">
      {/* Header */}
      <div className="flex items-center justify-between pb-4 border-b border-[#1F2937]">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-cyan-950/80 border border-cyan-800/80 flex items-center justify-center text-cyan-400">
            <UploadCloud className="w-4 h-4" />
          </div>
          <div>
            <h3 className="text-base font-bold font-mono text-white tracking-tight">
              Streamed Multipart Video Upload
            </h3>
            <p className="text-xs text-[#9CA3AF] font-mono mt-0.5">
              1 MB chunks • 500 MB max • Automated OpenCV validation
            </p>
          </div>
        </div>
        {onClose && (
          <button
            onClick={onClose}
            className="p-1 rounded-md text-[#9CA3AF] hover:text-white hover:bg-[#1F2937] transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        )}
      </div>

      <div className="mt-5 space-y-4">
        {/* Success View */}
        {uploadedVideo ? (
          <div className="p-4 rounded-lg bg-emerald-950/20 border border-emerald-800/50 space-y-3 font-mono text-xs">
            <div className="flex items-center gap-2 text-emerald-400 font-bold">
              <CheckCircle2 className="w-5 h-5 shrink-0" />
              <span>Video Successfully Ingested & Persisted</span>
            </div>
            <div className="space-y-1.5 pt-2 border-t border-emerald-900/50 text-[#D1D5DB]">
              <div className="flex justify-between">
                <span className="text-[#9CA3AF]">Video ID:</span>
                <span className="text-cyan-400 font-bold">{uploadedVideo.video_id}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-[#9CA3AF]">Filename:</span>
                <span className="text-white">{uploadedVideo.filename}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-[#9CA3AF]">Dimensions:</span>
                <span>
                  {uploadedVideo.width}x{uploadedVideo.height} @ {uploadedVideo.fps} FPS
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-[#9CA3AF]">Duration & Frames:</span>
                <span>
                  {formatDuration(uploadedVideo.duration_seconds)} ({uploadedVideo.frame_count} frames)
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-[#9CA3AF]">Stored Size:</span>
                <span>{formatBytes(uploadedVideo.file_size_bytes)}</span>
              </div>
            </div>

            <div className="pt-3 flex gap-2 justify-end">
              <Button variant="outline" size="sm" onClick={resetForm}>
                Upload Another Video
              </Button>
              {onClose && (
                <Button variant="primary" size="sm" onClick={onClose}>
                  Done
                </Button>
              )}
            </div>
          </div>
        ) : (
          <>
            {/* Drag & Drop Target Area */}
            <div
              onDragOver={onDragOver}
              onDragLeave={onDragLeave}
              onDrop={onDrop}
              onClick={() => fileInputRef.current?.click()}
              className={`border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-all ${
                isDragging
                  ? "border-cyan-400 bg-cyan-950/30"
                  : selectedFile
                  ? "border-cyan-800/80 bg-[#161F30]"
                  : "border-[#374151] hover:border-cyan-500/60 bg-[#0F172A]/40"
              }`}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept={ALLOWED_VIDEO_EXTENSIONS.join(",")}
                onChange={onFileInputChange}
                className="hidden"
                disabled={isUploading}
              />

              <div className="flex flex-col items-center gap-2 select-none">
                <div className="w-12 h-12 rounded-full bg-[#1A2234] border border-[#374151] flex items-center justify-center text-cyan-400">
                  <FileVideo className="w-6 h-6" />
                </div>
                {selectedFile ? (
                  <div className="space-y-1">
                    <p className="text-sm font-mono font-bold text-cyan-300">
                      {selectedFile.name}
                    </p>
                    <p className="text-xs font-mono text-[#9CA3AF]">
                      {formatBytes(selectedFile.size)} • Ready for upload
                    </p>
                  </div>
                ) : (
                  <div className="space-y-1">
                    <p className="text-sm font-medium text-[#F9FAFB]">
                      Drop raw surveillance footage here, or{" "}
                      <span className="text-cyan-400 underline font-semibold">browse</span>
                    </p>
                    <p className="text-xs font-mono text-[#6B7280]">
                      Supported formats: {ALLOWED_VIDEO_EXTENSIONS.join(", ")} (up to 500 MB)
                    </p>
                  </div>
                )}
              </div>
            </div>

            {/* Validation Error Banner */}
            {validationError && (
              <div className="p-3 rounded-lg bg-red-950/30 border border-red-800/60 flex items-start gap-2.5 text-xs font-mono text-red-300">
                <AlertTriangle className="w-4 h-4 text-red-400 shrink-0 mt-0.5" />
                <span>{validationError}</span>
              </div>
            )}

            {/* Server Error Banner */}
            {uploadError && (
              <div className="p-3 rounded-lg bg-red-950/40 border border-red-800/70 flex items-start gap-2.5 text-xs font-mono text-red-200">
                <AlertTriangle className="w-4 h-4 text-red-400 shrink-0 mt-0.5" />
                <div>
                  <div className="font-bold">Upload Failed</div>
                  <div className="text-[11px] mt-0.5 text-red-300">{uploadError}</div>
                </div>
              </div>
            )}

            {/* Upload Progress Bar */}
            {isUploading && (
              <div className="p-3 rounded-lg bg-[#161F30] border border-[#1F2937] space-y-2 font-mono text-xs">
                <div className="flex justify-between text-cyan-300">
                  <span>Streaming multipart payload...</span>
                  <span className="font-bold">{uploadProgress}%</span>
                </div>
                <div className="w-full h-2 bg-[#1F2937] rounded-full overflow-hidden">
                  <div
                    className="h-full bg-cyan-400 transition-all duration-150"
                    style={{ width: `${uploadProgress}%` }}
                  />
                </div>
                {uploadProgress >= 100 && (
                  <p className="text-[11px] text-cyan-400/80 animate-pulse">
                    Validating video integrity and extracting stream metadata with OpenCV...
                  </p>
                )}
              </div>
            )}

            {/* Quick Dev Action: Load Sample Fixture */}
            {APP_CONFIG.isDev && (
              <div className="p-3 rounded-lg bg-[#161F30]/60 border border-[#1F2937] flex items-center justify-between text-xs font-mono">
                <div>
                  <div className="flex items-center gap-1.5">
                    <span className="font-semibold text-white">Benchmark Sample</span>
                    <Badge variant="cyan" size="sm">DEV</Badge>
                  </div>
                  <p className="text-[11px] text-[#9CA3AF] mt-0.5">
                    MVI_20011_sample.mp4 (75 frames @ 25 FPS)
                  </p>
                </div>
                <Button
                  variant="secondary"
                  size="sm"
                  onClick={handleLoadSample}
                  disabled={isUploading || isLoadingSample}
                  icon={isLoadingSample ? <RotateCw className="w-3.5 h-3.5 animate-spin" /> : <Play className="w-3.5 h-3.5 text-cyan-400" />}
                >
                  {isLoadingSample ? "Loading..." : "Load Sample Clip"}
                </Button>
              </div>
            )}

            {/* Action Buttons */}
            <div className="pt-2 flex items-center justify-between">
              {selectedFile ? (
                <button
                  type="button"
                  onClick={resetForm}
                  disabled={isUploading}
                  className="text-xs font-mono text-[#9CA3AF] hover:text-white underline disabled:opacity-50"
                >
                  Clear Selection
                </button>
              ) : (
                <div />
              )}

              <div className="flex items-center gap-2">
                {onClose && (
                  <Button variant="outline" size="sm" onClick={onClose} disabled={isUploading}>
                    Cancel
                  </Button>
                )}
                <Button
                  variant="primary"
                  size="sm"
                  onClick={handleUpload}
                  disabled={!selectedFile || isUploading}
                  icon={isUploading ? <RotateCw className="w-3.5 h-3.5 animate-spin" /> : <UploadCloud className="w-3.5 h-3.5" />}
                >
                  {isUploading ? "Streaming Video..." : "Upload Video Stream"}
                </Button>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
};
