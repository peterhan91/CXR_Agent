import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "CXR Agent",
  description: "Chest X-ray agentic report generation with radiologist-in-the-loop",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className="bg-bg text-text-primary font-sans antialiased">
        {children}
      </body>
    </html>
  );
}
