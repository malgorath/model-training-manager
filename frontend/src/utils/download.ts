/**
 * Download utility functions for handling file downloads in the browser.
 */

/**
 * Triggers a file download in the browser from a Blob.
 * 
 * @param blob - The Blob containing the file data
 * @param filename - The desired filename for the download
 */
export function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

/**
 * Fetches a file from a URL and triggers a download.
 * 
 * @param url - The URL to fetch the file from
 * @param filename - The desired filename for the download
 * @returns Promise that resolves when download is initiated
 */
export async function downloadFromUrl(url: string, filename: string): Promise<void> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Download failed: ${response.statusText}`);
  }
  const blob = await response.blob();
  downloadBlob(blob, filename);
}

/**
 * Extracts filename from Content-Disposition header.
 * 
 * @param header - The Content-Disposition header value
 * @param fallback - Fallback filename if extraction fails
 * @returns The extracted or fallback filename
 */
export function getFilenameFromHeader(header: string | null, fallback: string): string {
  if (!header) return fallback;
  
  const filenameMatch = header.match(/filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/);
  if (filenameMatch && filenameMatch[1]) {
    return filenameMatch[1].replace(/['"]/g, '');
  }
  
  return fallback;
}

/**
 * Formats file size in human-readable format.
 * 
 * @param bytes - File size in bytes
 * @returns Formatted string (e.g., "1.5 MB")
 */
export function formatFileSize(bytes: number): string {
  if (bytes === 0) return '0 Bytes';
  
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(2))} ${sizes[i]}`;
}
