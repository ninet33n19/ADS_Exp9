"use client";

import React, { useState } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { DrawingCanvas } from "@/components/DrawingCanvas";
import { Brain, Loader2 } from "lucide-react";

interface PredictionResult {
  prediction: number;
  confidence: number;
  probabilities: number[];
}

function App() {
  const [imageData, setImageData] = useState<string>("");
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string>("");

  const handleDrawComplete = (data: string) => {
    setImageData(data);
    setPrediction(null);
    setError("");
  };

  const predictDigit = async () => {
    if (!imageData) {
      setError("Please draw a digit first");
      return;
    }

    setIsLoading(true);
    setError("");

    try {
      const response = await fetch("http://localhost:5000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ image: imageData }),
      });

      if (!response.ok) {
        throw new Error("Failed to get prediction");
      }

      const result = await response.json();
      setPrediction(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-8">
      <div className="max-w-6xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-800 mb-2 flex items-center justify-center gap-2">
            <Brain className="w-10 h-10 text-indigo-600" />
            MNIST Digit Recognition
          </h1>
          <p className="text-gray-600">
            Draw a digit and let AI predict what it is!
          </p>
        </div>

        <div className="grid md:grid-cols-2 gap-8">
          {/* Drawing Area */}
          <Card>
            <CardHeader>
              <CardTitle>Draw a Digit</CardTitle>
              <CardDescription>
                Use your mouse or touch to draw a single digit (0-9)
              </CardDescription>
            </CardHeader>
            <CardContent>
              <DrawingCanvas onDrawComplete={handleDrawComplete} />
              <Button
                onClick={predictDigit}
                disabled={isLoading || !imageData}
                className="w-full mt-4"
              >
                {isLoading ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Predicting...
                  </>
                ) : (
                  "Predict Digit"
                )}
              </Button>
              {error && (
                <div className="mt-4 p-3 bg-red-100 border border-red-400 text-red-700 rounded">
                  {error}
                </div>
              )}
            </CardContent>
          </Card>

          {/* Results Area */}
          <Card>
            <CardHeader>
              <CardTitle>Prediction Results</CardTitle>
              <CardDescription>
                AI models confidence for each digit
              </CardDescription>
            </CardHeader>
            <CardContent>
              {prediction ? (
                <div className="space-y-4">
                  <div className="text-center">
                    <div className="text-6xl font-bold text-indigo-600 mb-2">
                      {prediction.prediction}
                    </div>
                    <div className="text-sm text-gray-600">
                      Confidence: {(prediction.confidence * 100).toFixed(1)}%
                    </div>
                  </div>

                  <div className="space-y-2">
                    <h4 className="font-semibold text-sm text-gray-700">
                      All Probabilities:
                    </h4>
                    {prediction.probabilities.map((prob, index) => (
                      <div key={index} className="flex items-center space-x-2">
                        <span className="w-8 text-sm font-medium">
                          {index}:
                        </span>
                        <Progress value={prob * 100} className="flex-1" />
                        <span className="w-12 text-sm text-gray-600">
                          {(prob * 100).toFixed(1)}%
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="text-center text-gray-500 py-12">
                  <Brain className="w-16 h-16 mx-auto mb-4 text-gray-300" />
                  <p>Draw a digit and click predict to see results</p>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}

export default App;
