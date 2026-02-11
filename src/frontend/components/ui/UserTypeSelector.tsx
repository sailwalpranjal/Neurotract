'use client';

import { useState, useEffect } from 'react';
import { useAppStore } from '@/lib/store';
import { UserType } from '@/lib/types';

const USER_TYPES: { value: UserType; label: string; description: string }[] = [
  { value: 'doctor', label: 'Clinician', description: 'Advanced clinical metrics and findings' },
  { value: 'student', label: 'Researcher', description: 'Detailed explanations and methodology' },
  { value: 'general', label: 'Student', description: 'Simplified overviews and plain language' },
];

interface UserTypeSelectorProps {
  compact?: boolean;
}

export default function UserTypeSelector({ compact = false }: UserTypeSelectorProps) {
  const { userType, setUserType } = useAppStore();
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  if (compact) {
    return (
      <div className="flex gap-1 bg-white/5 rounded-lg p-1">
        {USER_TYPES.map((type) => (
          <button
            key={type.value}
            onClick={() => setUserType(type.value)}
            className={`px-3 py-1.5 text-xs rounded-md transition-all ${
              mounted && userType === type.value
                ? 'bg-primary-600 text-white font-medium'
                : 'text-gray-400 hover:text-gray-200 hover:bg-white/5'
            }`}
          >
            {type.label}
          </button>
        ))}
      </div>
    );
  }

  return (
    <div className="glass rounded-xl p-6">
      <h3 className="text-lg font-semibold mb-2">Viewing Mode</h3>
      <p className="text-sm text-gray-400 mb-4">
        Choose how information is presented across the application
      </p>
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        {USER_TYPES.map((type) => (
          <button
            key={type.value}
            onClick={() => setUserType(type.value)}
            className={`p-4 rounded-lg text-left transition-all ${
              mounted && userType === type.value
                ? 'bg-primary-600/20 border-2 border-primary-500/50'
                : 'bg-white/5 border-2 border-transparent hover:border-white/20'
            }`}
          >
            <div className="font-semibold mb-1">{type.label}</div>
            <div className="text-xs text-gray-400">{type.description}</div>
          </button>
        ))}
      </div>
    </div>
  );
}
