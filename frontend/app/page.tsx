import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-center gap-6 bg-background p-4">
      <Card className="w-full max-w-md">
        <CardHeader>
          <CardTitle className="text-2xl text-center">Welcome</CardTitle>
        </CardHeader>
        <CardContent className="flex flex-col space-y-4">
          <Link href="/mnist" passHref>
            <Button className="w-full">MNIST Page</Button>
          </Link>
          <Link href="/spam-detection" passHref>
            <Button className="w-full">Spam Detection Page</Button>
          </Link>
        </CardContent>
      </Card>
    </main>
  );
}
