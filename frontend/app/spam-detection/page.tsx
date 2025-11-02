"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Loader2, Shield, AlertTriangle, CheckCircle } from "lucide-react";

interface PredictionResult {
  text: string;
  is_spam: boolean;
  confidence: number;
  spam_probability: number;
  message: string;
}

export default function SpamDetector() {
  const [text, setText] = useState("");
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const detectSpam = async () => {
    if (!text.trim()) {
      setError("Please enter some text to analyze");
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch("http://localhost:5000/spam/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ text }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "Failed to analyze text");
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred");
    } finally {
      setLoading(false);
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return "bg-green-500";
    if (confidence >= 0.6) return "bg-yellow-500";
    return "bg-red-500";
  };

  const getSpamProbabilityColor = (probability: number) => {
    if (probability >= 0.7) return "text-red-600";
    if (probability >= 0.4) return "text-yellow-600";
    return "text-green-600";
  };

  return (
    <main className="min-h-screen bg-linear-to-br from-gray-50 to-gray-100 py-12">
      <div className="max-w-4xl mx-auto p-6 space-y-6">
        <div className="text-center space-y-2">
          <h1 className="text-4xl font-bold tracking-tight">
            Spam Detection System
          </h1>
          <p className="text-muted-foreground">
            Advanced NLP-powered spam detection for your messages
          </p>
        </div>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Shield className="h-5 w-5" />
              Analyze Text
            </CardTitle>
            <CardDescription>
              Enter a message below to check if its spam or legitimate
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <Textarea
              placeholder="Enter your message here..."
              value={text}
              onChange={(e) => setText(e.target.value)}
              className="min-h-[100px]"
            />
            <Button
              onClick={detectSpam}
              disabled={loading || !text.trim()}
              className="w-full"
            >
              {loading ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Analyzing...
                </>
              ) : (
                "Detect Spam"
              )}
            </Button>
          </CardContent>
        </Card>

        {error && (
          <Alert variant="destructive">
            <AlertTriangle className="h-4 w-4" />
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {result && (
          <Card
            className={result.is_spam ? "border-red-200" : "border-green-200"}
          >
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                {result.is_spam ? (
                  <>
                    <AlertTriangle className="h-5 w-5 text-red-600" />
                    <span className="text-red-600">Spam Detected!</span>
                  </>
                ) : (
                  <>
                    <CheckCircle className="h-5 w-5 text-green-600" />
                    <span className="text-green-600">Not Spam</span>
                  </>
                )}
              </CardTitle>
              <CardDescription>{result.message}</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <p className="text-sm font-medium mb-2">Analyzed Text:</p>
                <p className="text-sm bg-muted p-3 rounded-md">{result.text}</p>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-sm font-medium mb-2">Confidence Level:</p>
                  <div className="flex items-center gap-2">
                    <div className="flex-1 bg-gray-200 rounded-full h-2">
                      <div
                        className={`h-2 rounded-full ${getConfidenceColor(result.confidence)}`}
                        style={{ width: `${result.confidence * 100}%` }}
                      />
                    </div>
                    <span className="text-sm font-medium">
                      {(result.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>

                <div>
                  <p className="text-sm font-medium mb-2">Spam Probability:</p>
                  <p
                    className={`text-lg font-bold ${getSpamProbabilityColor(result.spam_probability)}`}
                  >
                    {(result.spam_probability * 100).toFixed(1)}%
                  </p>
                </div>
              </div>

              <div className="flex gap-2">
                <Badge variant={result.is_spam ? "destructive" : "default"}>
                  {result.is_spam ? "SPAM" : "HAM"}
                </Badge>
                <Badge variant="outline">
                  Confidence: {(result.confidence * 100).toFixed(1)}%
                </Badge>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </main>
  );
}
