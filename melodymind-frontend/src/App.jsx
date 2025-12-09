import React, { useState } from 'react';
import { Upload, Music, Disc, Activity, Zap, Radio } from 'lucide-react';
import { predict } from './api/predict';

export default function App() {
  const [file, setFile] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      setFile(e.target.files[0]);
      setPrediction(null);
      setError(null);
    }
  };

  const handleUpload = async () => {
    if (!file) return;

    setLoading(true);
    setError(null);

    try {
      // Call the API helper which handles FormData and fetch internally
      const data = await predict(file);

      // Artificial delay to show off the animation if the API is too fast
      setTimeout(() => {
        setPrediction(data);
        setLoading(false);
      }, 1500);
    } catch (err) {
      console.error(err);
      setError('Connection Error. Is the backend jamming? Check please');
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 flex items-center justify-center p-4 relative overflow-hidden font-sans">
      {/* Dynamic Background */}
      <div className="absolute inset-0 z-0 opacity-40">
        <div className="absolute top-0 -left-4 w-72 h-72 bg-purple-500 rounded-full mix-blend-multiply filter blur-xl animate-blob"></div>
        <div className="absolute top-0 -right-4 w-72 h-72 bg-yellow-500 rounded-full mix-blend-multiply filter blur-xl animate-blob animation-delay-2000"></div>
        <div className="absolute -bottom-8 left-20 w-72 h-72 bg-pink-500 rounded-full mix-blend-multiply filter blur-xl animate-blob animation-delay-4000"></div>
      </div>

      <div className="w-full max-w-md bg-white/10 backdrop-blur-lg border border-white/20 rounded-3xl shadow-2xl overflow-hidden z-10 text-white transition-all transform hover:scale-[1.01] duration-500">
        
        {/* Header Section */}
        <div className="p-8 bg-gradient-to-r from-violet-600 to-indigo-600 relative overflow-hidden">
          <div className="absolute top-0 right-0 p-4 opacity-20">
            <Music size={100} />
          </div>
          <h1 className="text-3xl font-black tracking-tighter mb-2 flex items-center gap-3">
            <Radio className="animate-pulse text-pink-300" />
            MelodyMind<span className="text-pink-400">AI</span>
          </h1>
          <p className="text-indigo-100 font-medium opacity-90">AI-Powered Genre Decoder</p>
        </div>

        <div className="p-8 space-y-8">
          
          {/* Main Visual / Interaction Area */}
          <div className="flex justify-center my-4">
             <div className={`relative w-40 h-40 rounded-full border-4 border-gray-800 shadow-2xl flex items-center justify-center bg-gray-900 transition-all duration-700 ${loading ? 'animate-spin' : ''}`}>
                {/* Vinyl Record Styling */}
                <div className="absolute inset-2 border border-gray-700 rounded-full opacity-50"></div>
                <div className="absolute inset-6 border border-gray-700 rounded-full opacity-50"></div>
                <div className="absolute inset-10 border border-gray-700 rounded-full opacity-50"></div>
                
                {/* Center Label */}
                <div className={`w-16 h-16 rounded-full flex items-center justify-center bg-gradient-to-tr from-pink-500 to-violet-500 z-10 shadow-inner ${prediction ? 'scale-110' : 'scale-100'} transition-transform`}>
                    {loading ? <Activity className="text-white" /> : <Disc className="text-white" />}
                </div>

                {/* Glow effect when active */}
                {loading && (
                   <div className="absolute -inset-4 bg-pink-500 rounded-full blur-xl opacity-30 animate-pulse"></div>
                )}
             </div>
          </div>

          {/* Upload Area */}
          {!prediction && !loading && (
            <div className="group relative">
              <input 
                type="file" 
                accept=".mp3,audio/*"
                onChange={handleFileChange} 
                className="hidden" 
                id="audio-upload"
              />
              <label 
                htmlFor="audio-upload" 
                className={`
                  block w-full p-6 border-2 border-dashed rounded-2xl text-center cursor-pointer transition-all duration-300
                  ${file 
                    ? 'border-pink-500 bg-pink-500/10' 
                    : 'border-gray-500 hover:border-violet-400 hover:bg-white/5'}
                `}
              >
                {/* Removed 'pointer-events-none' to ensure clicks register on the text too */}
                <div className="flex flex-col items-center gap-3">
                  <Upload className={`w-8 h-8 ${file ? 'text-pink-400' : 'text-gray-400 group-hover:text-violet-400'}`} />
                  <span className="font-bold text-lg">
                    {file ? file.name : "Click to Upload or Drop MP3"}
                  </span>
                  <span className="text-xs text-gray-400 uppercase tracking-widest">
                    {file ? "Ready to spin" : "Select a track"}
                  </span>
                </div>
              </label>
            </div>
          )}

          {/* Action Button */}
          {!prediction && (
            <button
              onClick={handleUpload}
              disabled={!file || loading}
              className={`
                w-full py-4 px-6 rounded-xl font-black text-lg uppercase tracking-wider transition-all duration-300 transform
                flex items-center justify-center gap-3 shadow-lg active:scale-95
                ${!file || loading 
                  ? 'bg-gray-700 text-gray-500 cursor-not-allowed' 
                  : 'bg-gradient-to-r from-pink-500 to-violet-600 hover:from-yellow-400 hover:to-yellow-500 hover:text-gray-900 hover:shadow-yellow-500/50 text-white hover:scale-105 shadow-pink-500/25'}
              `}
            >
              {loading ? 'Analyzing Waves...' : (
                <>
                  <Zap className="w-5 h-5 fill-current" /> Identify Genre
                </>
              )}
            </button>
          )}

          {/* Results Reveal */}
          {prediction && (
            <div className="animate-fade-in-up">
              <div className="bg-gradient-to-br from-gray-800 to-black border border-gray-700 rounded-2xl p-6 text-center relative overflow-hidden">
                <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-pink-500 via-purple-500 to-indigo-500"></div>
                
                <p className="text-gray-400 text-xs uppercase tracking-[0.2em] mb-2">Analysis Complete</p>
                <h2 className="text-5xl font-black text-transparent bg-clip-text bg-gradient-to-r from-pink-400 to-violet-400 mb-2">
                  {prediction.genre || "UNKNOWN"}
                </h2>
                
                {prediction.confidence && (
                  <div className="inline-block bg-gray-900 rounded-full px-4 py-1 border border-gray-700 mt-2">
                    <p className="text-xs font-mono text-gray-300">
                      CONFIDENCE: <span className="text-green-400">{(prediction.confidence * 100).toFixed(1)}%</span>
                    </p>
                  </div>
                )}

                <button 
                  onClick={() => {setPrediction(null); setFile(null);}}
                  className="mt-6 text-sm text-gray-500 hover:text-white underline underline-offset-4 decoration-pink-500"
                >
                  Analyze Another Track
                </button>
              </div>
            </div>
          )}
          
          {/* Error Message */}
          {error && (
            <div className="bg-red-500/20 border-l-4 border-red-500 p-4 rounded text-red-200 text-sm font-mono">
              ERROR: {error}
            </div>
          )}

        </div>
      </div>
      
      {/* Background CSS for blobs */}
      <style>{`
        @keyframes blob {
          0% { transform: translate(0px, 0px) scale(1); }
          33% { transform: translate(30px, -50px) scale(1.1); }
          66% { transform: translate(-20px, 20px) scale(0.9); }
          100% { transform: translate(0px, 0px) scale(1); }
        }
        .animate-blob {
          animation: blob 7s infinite;
        }
        .animation-delay-2000 {
          animation-delay: 2s;
        }
        .animation-delay-4000 {
          animation-delay: 4s;
        }
      `}</style>
    </div>
  );
}