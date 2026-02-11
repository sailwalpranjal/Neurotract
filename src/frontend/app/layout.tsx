import type { Metadata, Viewport } from 'next';
import { Inter } from 'next/font/google';
import '../styles/globals.css';
import Header from '@/components/ui/Header';
import Notifications from '@/components/ui/Notifications';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'NeuroTract - Brain Tractography & Connectomics',
  description: 'Advanced diffusion MRI tractography and brain network analysis platform',
  keywords: 'tractography, connectomics, brain imaging, MRI, diffusion imaging',
  authors: [{ name: 'NeuroTract Team' }],
};

export const viewport: Viewport = {
  width: 'device-width',
  initialScale: 1,
  themeColor: '#1a1a2e',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
        <meta charSet="utf-8" />
        <link rel="icon" href="/favicon.ico" />
      </head>
      <body className={inter.className}>
        <div className="min-h-screen flex flex-col">
          <Header />
          <main className="flex-1 flex flex-col">
            {children}
          </main>
          <Notifications />
        </div>
      </body>
    </html>
  );
}
